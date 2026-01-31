// static/js/app.js — smooth & anti-flicker

const video = document.getElementById('video');
const ov = document.getElementById('ov');
const cap = document.getElementById('cap');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnFlash = document.getElementById('btnFlash');
const elLbl = document.getElementById('lbl');
const elScr = document.getElementById('scr');
const elLat = document.getElementById('lat');
const modelname = document.getElementById('modelname');
const btnAnalysis = document.getElementById('btnAnalysis');
let analysisMode = false;

btnAnalysis.onclick = async () => {
  const r = await fetch('/toggle-analysis', { method: 'POST' });
  const j = await r.json();

  analysisMode = j.analysis_mode;

  if (analysisMode) {
    btnAnalysis.textContent = "⛔ Selesai Analisis";
    btnAnalysis.classList.add("analysis-on");
  } else {
    btnAnalysis.textContent = "📊 Mulai Analisis Harian";
    btnAnalysis.classList.remove("analysis-on");
  }
};

let running = false, busy = false, rafId = null, stream = null;

// === FLASH / TORCH ===
let videoTrack = null;
let flashOn = false;


// Target awal (adaptif)
let targetW = 640;      // resolusi kirim awal
let targetFPS = 12;     // FPS kirim awal
let lastSend = 0;
let rttEMA = 80;        // ms
const EMA_ALPHA = 0.2;  // smoothing rata-rata

// Anti-kedip: tahan overlay saat miss singkat + smoothing bbox
const HOLD_MS = 350;    // tahan overlay maksimal 350ms saat tidak ada deteksi
const SMOOTH_POS = 0.45; // 0..1 (semakin kecil = lebih halus)
const SMOOTH_SIZE = 0.35;

let lastBoxes = [];     // [{x,y,w,h,label,score}]
let lastDetTs = 0;
let lastLabel = '—';
let lastScore = null;

// Double buffer untuk overlay agar drawing atomik
const ovTmp = document.createElement('canvas');
const ctxOv = ov.getContext('2d');
let ctxTmp = null;

function now(){ return performance.now(); }

async function fetchModelInfo(){
  try{
    const r = await fetch('/modelinfo');
    const j = await r.json();
    modelname.textContent = `${j.classes?.length||0} kelas • ${j.model||'SVM'}`;
  }catch(e){ modelname.textContent = 'gagal memuat info'; }
}
fetchModelInfo();

// ====== Utility ======
function sizeCanvases(){
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  ov.width = vw; ov.height = vh;
  ovTmp.width = vw; ovTmp.height = vh;
  ctxTmp = ovTmp.getContext('2d');
}

function lerp(a, b, t){ return a + (b - a) * t; }

function smoothBox(prev, curr){
  if(!prev) return {...curr};
  return {
    x: Math.round(lerp(prev.x, curr.x, 1 - SMOOTH_POS)),
    y: Math.round(lerp(prev.y, curr.y, 1 - SMOOTH_POS)),
    w: Math.round(lerp(prev.w, curr.w, 1 - SMOOTH_SIZE)),
    h: Math.round(lerp(prev.h, curr.h, 1 - SMOOTH_SIZE)),
    label: curr.label ?? prev.label,
    score: typeof curr.score === 'number' ? curr.score : prev.score
  };
}

function drawBoxes(ctx, boxes){
  ctx.clearRect(0,0,ovTmp.width,ovTmp.height);
  ctx.lineWidth = 3;
  ctx.font = '16px system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
  ctx.textBaseline = 'top';
  ctx.strokeStyle = '#46c2ff';

  for(const b of boxes){
    const {x,y,w,h,label,score} = b;
    ctx.strokeRect(x,y,w,h);
    const text = `${label ?? lastLabel}${typeof score==='number' ? ' ' + score.toFixed(2) : ''}`;
    const pad=4, th=24, tw=ctx.measureText(text).width + pad*2;
    ctx.fillStyle='rgba(0,0,0,0.6)'; ctx.fillRect(x, y-(th+2), tw, th);
    ctx.fillStyle='#e9eef7'; ctx.fillText(text, x+pad, y-(th)+4);
  }
  // salin atomik ke overlay agar tidak terlihat “gambar bertahap”
  ctxOv.clearRect(0,0,ov.width,ov.height);
  ctxOv.drawImage(ovTmp, 0, 0);
}

// Base64 → Blob (untuk createImageBitmap)
function b64ToBlob(b64, type='image/jpeg'){
  const bin = atob(b64);
  const len = bin.length;
  const u8 = new Uint8Array(len);
  for(let i=0;i<len;i++) u8[i] = bin.charCodeAt(i);
  return new Blob([u8], {type});
}

// ====== Camera handling ======
async function startCam(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 }
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    // ===== SIMPAN TRACK KAMERA =====
    videoTrack = stream.getVideoTracks()[0];

    // ===== PASANG HANDLER FLASH =====
    btnFlash.onclick = async () => {
      const caps = videoTrack.getCapabilities();

      if (!caps.torch) {
        alert("Flash tidak didukung di perangkat ini");
        return;
      }

      flashOn = !flashOn;

      await videoTrack.applyConstraints({
        advanced: [{ torch: flashOn }]
      });

      btnFlash.textContent = flashOn ? "🔦 Flash ON" : "🔦 Flash OFF";
    };

    sizeCanvases();
    running = true;
    loop();

  }catch(e){
    alert('Gagal akses kamera: ' + e);
  }
}


function stopCam(){
  running = false;
   // 🔦 pastikan flash mati
  if (videoTrack) {
    try {
      videoTrack.applyConstraints({ advanced: [{ torch: false }] });
    } catch(e){}
  }
  if(rafId) cancelAnimationFrame(rafId);
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream = null; }
  ctxOv.clearRect(0,0,ov.width,ov.height);
  lastBoxes = [];
}

btnStart.onclick = startCam;
btnStop.onclick = stopCam;

// ====== Main loop with pacing ======
async function loop(){
  if(!running) return;
  const t = now();
  const minInterval = 1000/targetFPS;

  if(!busy && (t - lastSend) >= minInterval){
    lastSend = t;
    await sendFrame();
  }
  rafId = requestAnimationFrame(loop);
}

// ====== Send frame & render overlay ======
async function sendFrame(){
  busy = true;
  const t0 = now();

  const vw = video.videoWidth, vh = video.videoHeight;
  if(!vw || !vh){ busy=false; return; }

  // hitung skala capture sesuai targetW adaptif (hemat bandwidth)
  const scale = targetW / vw;
  const cw = Math.round(vw * scale);
  const ch = Math.round(vh * scale);
  if (cap.width !== cw || cap.height !== ch) { cap.width = cw; cap.height = ch; }

  const ctxCap = cap.getContext('2d');
  ctxCap.drawImage(video, 0, 0, cap.width, cap.height);

  const blob = await new Promise(res=>cap.toBlob(res, 'image/jpeg', 0.65)); // sedikit lebih kuat → lebih ringan

  // Timeout agar request lama tidak menumpuk
  const controller = new AbortController();
  const timeoutId = setTimeout(()=>controller.abort(), 4000);

  try{
    const r = await fetch('/infer', {
      method:'POST',
      headers:{'Content-Type':'application/octet-stream', 'X-Flash-On': flashOn ? '1' : '0'},
      body: blob,
      keepalive: true,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    const j = await r.json();

    let didDraw = false;

    if(j && j.ok){
      // Update stat
      if (typeof j.label !== 'undefined') { lastLabel = j.label ?? lastLabel; elLbl.textContent = lastLabel ?? '—'; }
      if (typeof j.score === 'number') { lastScore = j.score; elScr.textContent = j.score.toFixed(2); }

      // ==== MODE 1: server kirim boxes (paling stabil, no kedip) ====
      if (Array.isArray(j.boxes) && j.boxes.length){
        const sx = ov.width  / (j.input_w || cap.width);
        const sy = ov.height / (j.input_h || cap.height);
        const curr = j.boxes.map(b => ({
          x: (b.x||0)*sx, y:(b.y||0)*sy, w:(b.w||0)*sx, h:(b.h||0)*sy,
          label: b.label ?? j.label, score: (typeof b.score==='number'? b.score : j.score)
        }));

        // smoothing terhadap lastBoxes (cocokkan panjang)
        const smoothed = curr.map((c, i) => smoothBox(lastBoxes[i], c));
        drawBoxes(ctxTmp, smoothed);

        lastBoxes = smoothed;
        lastDetTs = now();
        didDraw = true;
      }
      // ==== MODE 2: server kirim JPEG overlay (j.frame) ====
      else if (j.frame){
        // Jangan clear overlay dulu—decode gambar dahulu lalu replace (hindari kedip)
        const blob = b64ToBlob(j.frame, 'image/jpeg');
        try{
          const bmp = await createImageBitmap(blob);
          // salin atomik via buffer
          ctxTmp.clearRect(0,0,ovTmp.width,ovTmp.height);
          ctxTmp.drawImage(bmp, 0, 0, ovTmp.width, ovTmp.height);
          ctxOv.clearRect(0,0,ov.width,ov.height);
          ctxOv.drawImage(ovTmp, 0, 0);
          lastDetTs = now();
          didDraw = true;
        }catch{
          // fallback ke Image()
          await new Promise((resolve)=>{
            const img = new Image();
            img.onload = () => {
              ctxTmp.clearRect(0,0,ovTmp.width,ovTmp.height);
              ctxTmp.drawImage(img, 0, 0, ovTmp.width, ovTmp.height);
              ctxOv.clearRect(0,0,ov.width,ov.height);
              ctxOv.drawImage(ovTmp, 0, 0);
              resolve();
            };
            img.src = 'data:image/jpeg;base64,' + j.frame;
          });
          lastDetTs = now();
          didDraw = true;
        }
      }
    }

    // ==== Anti-kedip: tahan overlay sementara saat miss ====
    if (!didDraw){
      const dt = now() - lastDetTs;
      if (dt <= HOLD_MS && lastBoxes.length){
        // redraw last boxes (tanpa update) agar overlay tidak hilang tiba-tiba
        drawBoxes(ctxTmp, lastBoxes);
        didDraw = true;
      } else if (dt > HOLD_MS){
        // sudah terlalu lama tidak ada deteksi → clear sekali
        ctxOv.clearRect(0,0,ov.width,ov.height);
        lastBoxes = [];
      }
    }

    // ==== Adaptasi kualitas berdasarkan latensi ====
    const rtt = now() - t0;
    rttEMA = (1-EMA_ALPHA)*rttEMA + EMA_ALPHA*rtt;
    elLat.textContent = Math.round(rtt);

    if (rttEMA > 180) {            // jaringan/CPU berat → turunkan beban
      if (targetW > 480) targetW -= 80;
      if (targetFPS > 10) targetFPS -= 1;
    } else if (rttEMA < 90) {      // longgar → naikkan pelan
      if (targetW < 960) targetW += 80;
      if (targetFPS < 15) targetFPS += 1;
    }
  }catch(e){
    // Abort = timeout, abaikan saja; lain-lain log
    if (e.name !== 'AbortError') console.error(e);
  }finally{
    clearTimeout(timeoutId);
    busy = false;
  }
}

// ====== FLASH BUTTON ======
if (btnFlash) {
  btnFlash.onclick = async () => {
    if (!videoTrack) {
      alert('Kamera belum aktif');
      return;
    }

    const caps = videoTrack.getCapabilities?.();
    if (!caps || !caps.torch) {
      alert('Flash tidak didukung di perangkat ini');
      return;
    }

    flashOn = !flashOn;

    try {
      await videoTrack.applyConstraints({
        advanced: [{ torch: flashOn }]
      });
      btnFlash.textContent = flashOn ? '🔦 Flash ON' : '🔦 Flash OFF';
    } catch (e) {
      console.error('Flash error:', e);
      alert('Gagal mengatur flash');
    }
  };
}

// ====== Events ======
video.addEventListener('loadedmetadata', sizeCanvases);
window.addEventListener('resize', () => {
  sizeCanvases();
  // jangan clear brutal saat resize—biarkan frame berikutnya mengganti
});

