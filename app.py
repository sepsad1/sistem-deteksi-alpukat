# app.py — Deteksi Kematangan Alpukat (kamera HP, real-time, stable, one-box, per-track)
# Patch ini menambahkan:
# - Veto glossy (kilap kuat → plastik/mouse/permukaan licin) DITINGKATKAN (votes += 2)
# - Veto boxiness (terlalu kotak atau memanjang ekstrem)
# - Tuning mudah via ENV

import os, time, base64
from typing import List, Tuple, Optional
from collections import deque

import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, get_flashed_messages

# ===== OpenCV speed (aman) =====
try:
    cv2.setUseOptimized(True)
    if hasattr(cv2, "getNumberOfCPUs") and hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(cv2.getNumberOfCPUs())
except Exception:
    pass

# ====== Auth via DB ======
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError

# ===================== Konfigurasi =====================
MODEL_PATH    = os.getenv("MODEL_PATH", "models/avocado_svm.joblib")
MAX_W         = int(os.getenv("MAX_W", "560"))
JPEG_QUALITY  = int(os.getenv("JPEG_QUALITY", "80"))
RETURN_MODE   = os.getenv("RETURN_MODE", "boxes").lower()  # "boxes" | "frame"

# ---------- Gate bentuk ----------
MIN_AREA_RATIO   = float(os.getenv("MIN_AREA_RATIO", "0.0032"))
MIN_SOLIDITY     = float(os.getenv("MIN_SOLIDITY", "0.80"))
CIRC_MIN         = float(os.getenv("CIRC_MIN", "0.25"))
CIRC_MAX         = float(os.getenv("CIRC_MAX", "0.92"))
AR_MIN           = float(os.getenv("AR_MIN", "0.55"))
AR_MAX           = float(os.getenv("AR_MAX", "2.60"))
MIN_EXTENT       = float(os.getenv("MIN_EXTENT", "0.55"))

# ---------- Gate warna (HSV) ----------
SAT_MIN          = float(os.getenv("SAT_MIN", "0.18"))
HUE_MIN          = int(os.getenv("HUE_MIN", "15"))
HUE_MAX          = int(os.getenv("HUE_MAX", "140"))
COLOR_MIN        = float(os.getenv("COLOR_MIN", "0.55"))
RED_MAX          = float(os.getenv("RED_MAX",   "0.35"))
RED_MAX_DARK     = float(os.getenv("RED_MAX_DARK", "0.55"))
YELLOW_MAX       = float(os.getenv("YELLOW_MAX","0.40"))  # sedikit dipersempit

# ---------- Anti-non-avocado ----------
EDGE_DENS_MAX    = float(os.getenv("EDGE_DENS_MAX", "0.10"))
DEFECTS_MAX      = int(os.getenv("DEFECTS_MAX", "4"))
RIPPLE_ERR_MAX   = float(os.getenv("RIPPLE_ERR_MAX", "0.08"))
TAPER_MAX        = float(os.getenv("TAPER_MAX", "0.95"))
ANTI_VOTES_NEED  = int(os.getenv("ANTI_VOTES_NEED", "3"))
EDGE_TOUCH_MARGIN= int(os.getenv("EDGE_TOUCH_MARGIN", "3"))

# ===== Tambahan anti-labu jepang (lebih ketat) =====
GREEN_DOM        = float(os.getenv("GREEN_DOM", "0.50"))   # dominasi hijau lebih tinggi
TEXVAR_MIN       = float(os.getenv("TEXVAR_MIN", "22.0"))  # tekstur minimum (alpukat lebih kasar)

# ---------- Veto Alpukat Busuk ----------
ROTTEN_DARK_MIN = float(os.getenv("ROTTEN_DARK_MIN", "0.60"))
ROTTEN_TEX_MAX  = float(os.getenv("ROTTEN_TEX_MAX", "14.0"))
ROTTEN_RED_MAX  = float(os.getenv("ROTTEN_RED_MAX", "0.18"))

# ---- Veto tambahan (glossy & boxy) ----
GLOSS_S_MAX      = float(os.getenv("GLOSS_S_MAX", "0.18"))   # saturasi sangat rendah (highlight)
GLOSS_V_MIN      = float(os.getenv("GLOSS_V_MIN", "0.78"))   # value tinggi (kilap)
GLOSS_AREA_FRAC  = float(os.getenv("GLOSS_AREA_FRAC", "0.06"))
BOXY_IOU_MINRECT = float(os.getenv("BOXY_IOU_MINRECT", "0.82"))  # kontur vs minAreaRect terlalu kotak
BOXY_AR_MAX      = float(os.getenv("BOXY_AR_MAX", "3.6"))        # terlalu memanjang → cenderung bukan alpukat

# ---------- Negatif (label BUKAN ALPUKAT) ----------
NEGATIVE_MODE    = int(os.getenv("NEGATIVE_MODE", "1"))   # 1=aktif
NEG_LABEL        = os.getenv("NEG_LABEL", "BUKAN ALPUKAT")
NEG_MIN_AREA     = int(os.getenv("NEG_MIN_AREA", "1400"))

# ---------- Klasifikasi / fusi matang ----------
MIN_PROBA        = float(os.getenv("MIN_PROBA", "0.60"))
RIPE_ON_THRESH   = float(os.getenv("RIPE_ON_THRESH", "0.78"))
RIPE_OFF_THRESH  = float(os.getenv("RIPE_OFF_THRESH","0.42"))
EMA_ALPHA_RIPE   = float(os.getenv("EMA_ALPHA_RIPE", "0.35"))

# Stabilizer kematangan (per-track)
LOCK_MS               = int(os.getenv("LOCK_MS", "1200"))
SWITCH_CONFIRM_FRAMES = int(os.getenv("SWITCH_CONFIRM_FRAMES", "4"))

# ---------- Ukuran ----------
PX_PER_CM              = float(os.getenv("PX_PER_CM", "0"))
SIZE_CM_SMALL_MAX      = float(os.getenv("SIZE_CM_SMALL_MAX", "8.8"))
SIZE_CM_BIG_MIN        = float(os.getenv("SIZE_CM_BIG_MIN",   "9.6"))
SIZE_SMALL_MAX         = float(os.getenv("SIZE_SMALL_MAX", "0.27"))
SIZE_BIG_MIN           = float(os.getenv("SIZE_BIG_MIN",   "0.31"))
SIZE_SNAP_DELTA        = float(os.getenv("SIZE_SNAP_DELTA","0.15"))
EMA_ALPHA_SIZE         = float(os.getenv("EMA_ALPHA_SIZE", "0.35"))

# ---------- Toggles ----------
AUTO_CLAHE             = int(os.getenv("AUTO_CLAHE", "1"))
USE_ELLIPSE            = int(os.getenv("USE_ELLIPSE", "1"))
FALLBACK_DARK_ON_WHITE = int(os.getenv("FALLBACK_DARK_ON_WHITE", "0")) #kalo ngak bisa detek ubah jadi 1
TOPK_BOXES             = int(os.getenv("TOPK_BOXES", "3"))
USE_SVM                = int(os.getenv("USE_SVM", "0"))   # 0=skip svm (lebih cepat), 1=pakai svm


# ---------- Anti-kedap-kedip / Tracking ----------
TRACK_HOLD_MS      = int(os.getenv("TRACK_HOLD_MS", "900"))
TRACK_DROP_MS      = int(os.getenv("TRACK_DROP_MS", "1500"))
TRACK_MATCH_IOU    = float(os.getenv("TRACK_MATCH_IOU", "0.30"))
TRACK_MAX_MISSES   = int(os.getenv("TRACK_MAX_MISSES", "6"))

# ------- Proteksi “ikut ukuran sebelumnya” (switch guard) -------
SIZE_RESET_RATIO       = float(os.getenv("SIZE_RESET_RATIO", "0.28"))   # |major_now - major_track| / min(...) > ambang
SIZE_RESET_UP          = float(os.getenv("SIZE_RESET_UP", "0.22"))      # (major_now - major_track)/max(prev,1) > ambang
AREA_RATIO_RESET_ABS   = float(os.getenv("AREA_RATIO_RESET_ABS", "0.08"))
CENTER_DIST_RATIO_MAX  = float(os.getenv("CENTER_DIST_RATIO_MAX", "0.22")) # dist(center)/min(H,W)
STRICT_SWITCH_IOU_MAX  = float(os.getenv("STRICT_SWITCH_IOU_MAX", "0.22")) # IoU kecil + lonjakan → new track
SIZE_EMA_RESET_UP      = float(os.getenv("SIZE_EMA_RESET_UP", "0.28"))  # ema_target > size_ema*(1+X) → new track

# ------- Mode satu kotak + NMS ----------
ONE_BOX_MODE           = int(os.getenv("ONE_BOX_MODE", "1"))             # 1=aktif
NMS_IOU_THRESH         = float(os.getenv("NMS_IOU_THRESH", "0.45"))

# ------- Hold & prioritas tampil (anti-kedip khusus kecil) -------
HOLD_BONUS_SMALL_MS   = int(os.getenv("HOLD_BONUS_SMALL_MS", "700"))   # bonus hold utk size 'kecil'
TRACK_MIN_VISIBLE_MS  = int(os.getenv("TRACK_MIN_VISIBLE_MS", "500"))  # minimal tampil setelah track baru dibuat

# ===================== Load Model ======================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}. Jalankan train_svm.py dulu.")
artifact  = joblib.load(MODEL_PATH)
PIPELINE  = artifact["pipeline"]
CLASSES: List[str] = list(artifact.get("classes", []))

def get_final_classifier_and_classes(pipeline):
    try:
        if hasattr(pipeline, "steps"):
            for _, step in reversed(pipeline.steps):
                if hasattr(step, "classes_"):
                    return step, step.classes_
        if hasattr(pipeline, "classes_"):
            return pipeline, pipeline.classes_
    except Exception:
        pass
    return None, None

CLF, CLF_CLASSES = get_final_classifier_and_classes(PIPELINE)

def label_from_idx(idx: int) -> str:
    if CLF_CLASSES is not None and 0 <= idx < len(CLF_CLASSES):
        val = CLF_CLASSES[idx]
        if isinstance(val, (str, np.str_)): return str(val)
        try:
            ival = int(val)
            if CLASSES and 0 <= ival < len(CLASSES): return str(CLASSES[ival])
            return str(ival)
        except Exception:
            return str(val)
    if CLASSES and 0 <= idx < len(CLASSES): return str(CLASSES[idx])
    return str(idx)

def label_from_pred_value(pred_val) -> str:
    if isinstance(pred_val, (str, np.str_)): return str(pred_val)
    try:
        ival = int(pred_val)
        if CLASSES and 0 <= ival < len(CLASSES): return str(CLASSES[ival])
        return str(ival)
    except Exception:
        return str(pred_val)

print("[INFO] RETURN_MODE:", RETURN_MODE)

# ===================== Utils gambar ====================
def put_text_with_bg(img, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6,
                     txt_color=(255,255,255), bg_color=(0,0,0), thickness=1, pad=4, alpha=0.7):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = max(0, x - pad), max(0, y + pad)
    x2, y2 = min(img.shape[1]-1, x + tw + 2*pad), max(0, y - th - 2*pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y2), (x2, y1), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, scale, txt_color, thickness, cv2.LINE_AA)

def draw_box_with_label(img, box, label, score):
    x, y, w, h = box
    ll = label.lower()
    if "bukan alpukat" in ll:
        color = (40, 40, 220)   # warna khusus untuk negatif
    else:
        color = (60, 200, 90) if "matang" in ll else (60, 120, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    put_text_with_bg(img, f"{label} {score:.2f}", x + 4, max(24, y - 6))

# ===================== Prepro kamera HP =================
def gray_world(img_bgr):
    img = img_bgr.astype(np.float32)
    avgB, avgG, avgR = img.mean(axis=(0,1))
    avg = (avgB + avgG + avgR) / 3.0 + 1e-6
    img[...,0] *= avg/avgB; img[...,1] *= avg/avgG; img[...,2] *= avg/avgR
    return np.clip(img, 0, 255).astype(np.uint8)

def maybe_clahe_hsv(hsv):
    if not AUTO_CLAHE:
        return hsv
    v = hsv[:,:,2]
    m, s = cv2.meanStdDev(v)
    v_mean, v_std = float(m[0][0]), float(s[0][0])
    if v_mean < 70 or v_mean > 210 or v_std < 28:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v2 = clahe.apply(v)
        hsv = hsv.copy(); hsv[:,:,2] = v2
    return hsv

# ===================== Mask & rules ====================
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

def build_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K3, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K3, 1)
    return mask

def color_masks(hsv):
    red1   = build_mask(hsv, (  0, 90,  30), ( 10,255,255))
    red2   = build_mask(hsv, (170, 90,  30), (180,255,255))
    red    = cv2.bitwise_or(red1, red2)
    yellow = build_mask(hsv, ( 15, 40,  35), ( 38,255,255))
    green  = build_mask(hsv, ( 35, 35,  30), ( 90,255,255))
    olive  = build_mask(hsv, ( 20, 30,  25), ( 50,200,160))
    dark   = build_mask(hsv, (  0, 15,   0), (180,110, 85))
    return red, yellow, green, olive, dark

def avocado_color_rule(hsv, blob_mask, masks) -> Tuple[bool, dict]:
    red, yellow, green, olive, dark = masks
    total = float(cv2.countNonZero(blob_mask))
    if total <= 0:
        return False, {}
    def r(m): return cv2.countNonZero(cv2.bitwise_and(m, blob_mask)) / total
    r_red, r_yel, r_grn, r_olv, r_drk = r(red), r(yellow), r(green), r(olive), r(dark)
    s_mean = cv2.mean(hsv[:,:,1], mask=blob_mask)[0] / 255.0
    v_mean = cv2.mean(hsv[:,:,2], mask=blob_mask)[0] / 255.0
    h_mean = cv2.mean(hsv[:,:,0], mask=blob_mask)[0]

    bg_dark_veto = (r_drk >= 0.80 and r_grn <= 0.10 and r_olv <= 0.06 and s_mean < 0.25)
    # === VETO BENDA GELAP TOTAL (HP, mouse, bayangan) ===
    pure_dark_veto = (
        r_drk >= 0.80 and
        r_grn < 0.10 and
        r_olv < 0.10 and
        s_mean < 0.25
    )

    if pure_dark_veto:
        return False, {}

    labu_veto = (
    r_grn >= 0.38 and
    (r_olv + r_drk) <= 0.26 and
    v_mean >= 0.50 and
    s_mean >= 0.30
)


    signature = (r_grn + r_olv + r_drk + 0.4*r_yel)
    darkish   = (r_drk >= 0.30) or (v_mean <= 0.35)
    red_cap   = RED_MAX_DARK if darkish else RED_MAX
    purpleish = (r_red >= 0.15 and (h_mean < 20 or h_mean > 165))

    ok = (
        (signature >= COLOR_MIN and r_red <= red_cap and r_yel <= YELLOW_MAX and s_mean >= SAT_MIN and (HUE_MIN <= h_mean <= HUE_MAX or r_drk > 0.35))
        or (darkish and purpleish)
    )
    ok = ok and (not labu_veto) and (not bg_dark_veto)

    info = {"red": r_red, "yellow": r_yel, "green": r_grn, "olive": r_olv, "dark": r_drk,
            "sig": signature, "s_mean": s_mean, "v_mean": v_mean, "h_mean": h_mean,
            "red_cap": red_cap, "darkish": float(darkish)}
    return ok, info


def avocado_shape_rule(cnt, frame_hw) -> Tuple[bool, dict]:
    H, W = frame_hw
    area = cv2.contourArea(cnt)
    if area < 700 or (area / (H * W)) < MIN_AREA_RATIO:
        return False, {"reason": "area_small"}
    x, y, w, h = cv2.boundingRect(cnt)
    if x <= EDGE_TOUCH_MARGIN or y <= EDGE_TOUCH_MARGIN or (x + w) >= (W - EDGE_TOUCH_MARGIN) or (y + h) >= (H - EDGE_TOUCH_MARGIN):
        return False, {"reason": "touch_border"}
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    perim = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perim * perim) if perim > 0 else 0.0
    ar = w / float(h) if h > 0 else 1.0
    extent = area / float(w*h) if w*h > 0 else 0.0
    ok = (solidity >= MIN_SOLIDITY) and (CIRC_MIN <= circularity <= CIRC_MAX) and (AR_MIN <= ar <= AR_MAX) and (extent >= MIN_EXTENT)
    info = {"solidity": solidity, "circ": circularity, "ar": ar, "extent": extent, "area": area}
    return ok, info

# -------- Anti-non-avocado checks --------
def compute_edge_density(img_bgr, blob):
    ys, xs = np.where(blob > 0)
    if len(xs) < 50:
        return 0.0
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    roi = img_bgr[y1:y2+1, x1:x2+1]
    m = blob[y1:y2+1, x1:x2+1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.bitwise_and(edges, edges, mask=m)
    return float(cv2.countNonZero(edges)) / float(cv2.countNonZero(m) + 1e-6)

def count_convexity_defects(cnt):
    if len(cnt) < 5:
        return 0
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        return 0
    area = max(cv2.contourArea(cnt), 1.0)
    scale = max(8.0, np.sqrt(area) * 0.03)
    k = 0
    for i in range(defects.shape[0]):
        _, _, _, d = defects[i, 0]
        if d/256.0 > scale:
            k += 1
    return k

def ellipse_ripple_error(cnt, frame_hw):
    if len(cnt) < 5:
        return 0.0
    try:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
    except Exception:
        return 0.0
    H, W = frame_hw
    mask_cnt = np.zeros((H,W), np.uint8); cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
    mask_el  = np.zeros((H,W), np.uint8)
    cv2.ellipse(mask_el, (int(cx),int(cy)), (int(MA/2),int(ma/2)), angle, 0, 360, 255, -1)
    inter = cv2.countNonZero(cv2.bitwise_and(mask_cnt, mask_el))
    union = cv2.countNonZero(cv2.bitwise_or(mask_cnt, mask_el)) + 1e-6
    iou_val = inter / union
    return 1.0 - iou_val

def pear_taper_ratio(blob):
    ys, xs = np.where(blob > 0)
    if len(xs) < 50:
        return 1.0
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h = y2 - y1 + 1
    y_top = y1 + int(0.15*h)
    y_mid = y1 + int(0.50*h)
    def width_at(y):
        row = np.where(blob[y] > 0)[0]
        return int(row.max()-row.min()+1) if len(row) else 0
    wt = width_at(y_top); wm = width_at(y_mid)
    if wm == 0: return 1.0
    return float(wt) / float(wm)

def texture_variance(img_bgr, blob):
    ys, xs = np.where(blob > 0)
    if len(xs) < 50:
        return 0.0
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    roi = img_bgr[y1:y2+1, x1:x2+1]
    m = blob[y1:y2+1, x1:x2+1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    var = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
    active = max(cv2.countNonZero(m), 1)
    return var * 100.0 / active

# ---- Veto tambahan (fungsi helper) ----
def glossy_highlight_fraction(hsv, blob):
    """Fraksi area highlight mengkilap: S rendah, V tinggi → indikasi plastik/logam/mouse."""
    s = hsv[:,:,1].astype(np.float32)/255.0
    v = hsv[:,:,2].astype(np.float32)/255.0
    m_gloss = np.uint8((s <= GLOSS_S_MAX) & (v >= GLOSS_V_MIN)) * 255
    m_gloss = cv2.bitwise_and(m_gloss, m_gloss, mask=blob)
    g = float(cv2.countNonZero(m_gloss))
    t = float(cv2.countNonZero(blob) + 1e-6)
    return g / t

def boxiness_metrics(cnt, frame_hw):
    """Seberapa 'kotak' kontur: IoU dengan minAreaRect + aspect ratio ekstrem."""
    if len(cnt) < 5:
        return 0.0, 1.0
    try:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if w < 1 or h < 1:
            return 0.0, 1.0
        box = cv2.boxPoints(rect).astype(np.int32)

        H, W = frame_hw
        mc = np.zeros((H,W), np.uint8); cv2.drawContours(mc, [cnt], -1, 255, -1)
        mr = np.zeros((H,W), np.uint8); cv2.drawContours(mr, [box], -1, 255, -1)

        inter = cv2.countNonZero(cv2.bitwise_and(mc, mr))
        union = cv2.countNonZero(cv2.bitwise_or(mc, mr)) + 1e-6
        iou_minrect = inter / union
        ar = max(w,h) / max(1.0, min(w,h))
        return float(iou_minrect), float(ar)
    except Exception:
        return 0.0, 1.0

def is_non_avocado(img_bgr, hsv, cnt, frame_hw, flash_on=False):
    """Kembalikan (is_non, score, reasons) supaya bisa dipakai untuk NEGATIVE_MODE."""
    H, W = frame_hw
    blob = np.zeros((H,W), np.uint8)
    cv2.drawContours(blob, [cnt], -1, 255, -1)

    votes = 0
    reasons = []

    ed = compute_edge_density(img_bgr, blob)
    if ed >= EDGE_DENS_MAX:
        votes += 1; reasons.append(("edge_dens", ed))

    defects = count_convexity_defects(cnt)
    if defects >= DEFECTS_MAX:
        votes += 1; reasons.append(("defects", defects))

    rip = ellipse_ripple_error(cnt, frame_hw)
    if rip >= RIPPLE_ERR_MAX:
        votes += 1; reasons.append(("ripple", rip))

    taper = pear_taper_ratio(blob)
    if taper >= TAPER_MAX:
        votes += 1; reasons.append(("taper", taper))

    red, yellow, green, olive, dark = color_masks(hsv)
    total = float(cv2.countNonZero(blob)) + 1e-6
    r_grn = cv2.countNonZero(cv2.bitwise_and(green, blob)) / total
    r_olv = cv2.countNonZero(cv2.bitwise_and(olive, blob)) / total
    r_drk = cv2.countNonZero(cv2.bitwise_and(dark,  blob)) / total
    r_red = cv2.countNonZero(cv2.bitwise_and(red, blob)) / total


    tex   = texture_variance(img_bgr, blob)
    if (r_grn >= GREEN_DOM and (r_olv + r_drk) <= 0.24 and tex < TEXVAR_MIN):
        votes += 3; reasons.append(("green_dom_smooth", (r_grn, r_olv+r_drk, tex)))
    # === VETO ALPUKAT BUSUK / TIDAK LAYAK KONSUMSI ===
    if (
        r_drk >= ROTTEN_DARK_MIN and
        tex <= ROTTEN_TEX_MAX and
        r_red <= ROTTEN_RED_MAX
    ):
        votes += 3
        reasons.append(("rotten_avocado", (r_drk, tex, r_red)))

    # --- Veto glossy (kilap kuat = bukan kulit alpukat) ---
    gloss_thresh = GLOSS_AREA_FRAC
    if flash_on:
        gloss_thresh += 0.03   # toleransi kilap saat flash ON

    gfrac = glossy_highlight_fraction(hsv, blob)
    if gfrac >= gloss_thresh:
        votes += 3
        reasons.append(("glossy", gfrac))


    # --- Veto boxy (terlalu kotak/memanjang) ---
    iou_rect, ar_rect = boxiness_metrics(cnt, frame_hw)
    if (iou_rect >= BOXY_IOU_MINRECT) or (ar_rect >= BOXY_AR_MAX):
        votes += 1; reasons.append(("boxy", (iou_rect, ar_rect)))
        
    # --- Veto khusus labu siam (oval licin & simetris) ---
    if taper >= 0.75 and tex < TEXVAR_MIN and r_grn > 0.40:
        votes += 2
        reasons.append(("labu_shape_texture", (taper, tex)))


    is_non = (votes >= ANTI_VOTES_NEED)
    score = min(1.0, 0.20*votes + 0.30*max(0.0, r_grn - 0.40) + 0.30*max(0.0, (TEXVAR_MIN - tex)/TEXVAR_MIN))
    return is_non, float(score), reasons

# ===================== Fitur ===========================
def feature_vector(hsv, blob_mask, cnt, masks) -> np.ndarray:
    H, W = hsv.shape[:2]
    hist = cv2.calcHist([hsv], [0], blob_mask, [16], [0, 180]).astype(np.float32).flatten()
    hist = hist / (hist.sum() + 1e-8)
    s_mean = cv2.mean(hsv[:, :, 1], mask=blob_mask)[0] / 255.0
    v_mean = cv2.mean(hsv[:, :, 2], mask=blob_mask)[0] / 255.0
    total = cv2.countNonZero(blob_mask)
    rs = [cv2.countNonZero(cv2.bitwise_and(m, blob_mask)) / total if total > 0 else 0.0 for m in masks]
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    area_ratio  = area / float(H * W)
    bbox_ratio  = (w * h) / float(H * W)
    major       = max(w, h)
    if USE_ELLIPSE and len(cnt) >= 5:
        try:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            major = max(MA, ma)
        except Exception:
            pass
    major_axis_ratio = major / float(min(H, W))
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity    = area / hull_area if hull_area > 0 else 0.0
    perim       = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perim * perim) if perim > 0 else 0.0
    aspect_ratio= w / float(h) if h > 0 else 1.0
    return np.concatenate([
        hist,
        np.array([s_mean, v_mean], np.float32),
        np.array(rs, np.float32),
        np.array([area_ratio, bbox_ratio, major_axis_ratio, solidity, circularity, aspect_ratio], np.float32),
    ])

# ===================== Helper ukuran & NMS =================
def iou(b1, b2):
    x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
    xa = max(x1, x2); ya = max(y1, y2)
    xb = min(x1+w1, x2+w2); yb = min(y1+h1, y2+h2)
    iw = max(0, xb - xa); ih = max(0, yb - ya)
    inter = iw * ih
    if inter <= 0: return 0.0
    union = (w1*h1) + (w2*h2) - inter
    if union <= 0: return 0.0
    return inter / union

def box_center_xy(b):
    x, y, w, h = b
    return (x + w * 0.5, y + h * 0.5)

def nms_boxes(boxes, iou_thresh):
    if not boxes: return []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].get("score", 1.0), reverse=True)
    keep, used = [], [False]*len(boxes)
    def iou_box(i, j):
        bi, bj = boxes[i], boxes[j]
        return iou((bi["x"],bi["y"],bi["w"],bi["h"]), (bj["x"],bj["y"],bj["w"],bj["h"]))
    for i in order:
        if used[i]: continue
        keep.append(boxes[i])
        for j in order:
            if used[j] or j == i: continue
            if iou_box(i, j) >= iou_thresh:
                used[j] = True
        used[i] = True
    return keep

def size_metric_from(major_px: float, area_ratio_contour: float, frame_hw: Tuple[int,int]) -> float:
    H, W = frame_hw
    r_major = major_px / float(min(H, W))
    r_area  = np.sqrt(max(0.0, area_ratio_contour))
    return 0.55*r_major + 0.45*r_area if PX_PER_CM <= 0 else (major_px / PX_PER_CM)

def format_label(ripe_state: str, size_state: str) -> str:
    return f"{'Matang' if ripe_state=='matang' else 'Belum Matang'} berukuran {'Besar' if size_state=='besar' else 'Kecil'}"

import datetime

def save_daily_ripeness(track: dict):
    today = datetime.date.today()

    exists = AvocadoHistory.query.filter_by(
        track_id=track["id"],
        date=today
    ).first()
    if exists:
        return

    rec = AvocadoHistory(
        track_id=track["id"],
        size=track.get("size_state"),
        ripe_label=track.get("label"),      # 4 kategori tetap
        ripe_score=float(track.get("ripe_ema", 0.0)),
        date=today
    )
    if not ANALYSIS_MODE:
        return
    db.session.add(rec)
    db.session.commit()

def explain_fixed_category(label: str, color_info: dict):
    exp = []
    if "belum matang" in label.lower():
        exp += ["Warna hijau dominan", "Kulit belum menggelap"]
    else:
        exp += ["Warna kulit lebih gelap", "Tekstur lebih kasar"]

    if "kecil" in label.lower():
        exp.append("Ukuran buah kecil")
    else:
        exp.append("Ukuran buah besar")

    return {
        "kategori": label,
        "penjelasan": exp,
        "indikator": {
            "green": round(color_info.get("green", 0), 2),
            "dark": round(color_info.get("dark", 0), 2)
        }
    }

# ===================== Per-track smoothing =================
def decide_size_track(track: dict, major_px: float, area_ratio_contour: float, frame_hw: Tuple[int,int]) -> Tuple[str, float]:
    H, W = frame_hw
    r_major = major_px / float(min(H, W))
    r_area  = np.sqrt(max(0.0, area_ratio_contour))
    metric_ratio = 0.55*r_major + 0.45*r_area

    if PX_PER_CM > 0:
        ema_target = major_px / PX_PER_CM
        small_hi, big_lo = SIZE_CM_SMALL_MAX, SIZE_CM_BIG_MIN
    else:
        ema_target = metric_ratio
        small_hi, big_lo = SIZE_SMALL_MAX, SIZE_BIG_MIN

    if track.get("size_ema") is None:
        track["size_ema"] = ema_target
    if abs(ema_target - track["size_ema"]) >= SIZE_SNAP_DELTA:
        track["size_ema"] = 0.85*ema_target + 0.15*track["size_ema"]
    else:
        track["size_ema"] = (1-EMA_ALPHA_SIZE)*track["size_ema"] + EMA_ALPHA_SIZE*ema_target

    if track.get("size_state") is None:
        track["size_state"] = "besar" if track["size_ema"] >= (big_lo + small_hi)/2 else "kecil"
    else:
        if track["size_state"] == "besar" and track["size_ema"] < small_hi:
            track["size_state"] = "kecil"
        elif track["size_state"] == "kecil" and track["size_ema"] > big_lo:
            track["size_state"] = "besar"

    return track["size_state"], float(track["size_ema"])

def fused_ripeness_track(track: dict, probs: Optional[np.ndarray], color_info: dict, top_label: str) -> Tuple[str, float]:
    r_red = color_info.get("red", 0.0); r_yel = color_info.get("yellow", 0.0)
    r_grn = color_info.get("green", 0.0); r_olv = color_info.get("olive", 0.0); r_drk = color_info.get("dark", 0.0)
    s_mean = color_info.get("s_mean", 0.0); h_mean = color_info.get("h_mean", 60.0)

    color_score = np.clip(0.45*r_drk + 0.35*r_olv + 0.10*r_yel - 0.70*r_grn - 0.55*r_red + 0.50, 0, 1)

    model_score = 0.5
    if probs is not None and CLF_CLASSES is not None and len(CLF_CLASSES) == len(probs):
        mat_idx = [i for i,lab in enumerate(CLF_CLASSES) if isinstance(lab,(str,np.str_)) and ("matang" in str(lab).lower() or "ripe" in str(lab).lower())]
        bel_idx = [i for i,lab in enumerate(CLF_CLASSES) if isinstance(lab,(str,np.str_)) and ("belum" in str(lab).lower() or "mentah" in str(lab).lower() or "unripe" in str(lab).lower())]
        ps_m = float(probs[mat_idx].sum()) if len(mat_idx) else 0.0
        ps_b = float(probs[bel_idx].sum()) if len(bel_idx) else 0.0
        model_score = ps_m / (ps_m + ps_b) if (ps_m + ps_b) > 0 else (0.75 if ("matang" in str(top_label).lower()) else 0.25)

    fused = 0.35*model_score + 0.65*color_score

    green_veto = (
        (r_grn >= 0.40 and (r_olv + r_drk) <= 0.30) or
        (r_grn >= 0.35 and r_yel <= 0.12 and r_drk <= 0.22) or
        (h_mean >= 50 and s_mean >= 0.28 and r_grn >= 0.33)
    )
    if green_veto:
        fused = 0.05

    ripe_strong = (
        (r_drk >= 0.55 and r_grn <= 0.26) or
        (r_olv >= 0.28 and r_grn <= 0.22) or
        (r_red >= 0.12 and r_drk >= 0.35)
    )
    if ripe_strong and not green_veto:
        fused = max(fused, 0.88)

    if track.get("ripe_ema") is None:
        track["ripe_ema"] = fused
    track["ripe_ema"] = (1-EMA_ALPHA_RIPE)*track["ripe_ema"] + EMA_ALPHA_RIPE*fused

    if "ripe_votes" not in track:
        track["ripe_votes"] = deque(maxlen=8)
    track["ripe_votes"].append(1 if fused >= 0.5 else 0)
    votes_sum = sum(track["ripe_votes"])
    want_state = "matang" if votes_sum >= len(track["ripe_votes"])/2 else "belum matang"

    now_ms = int(time.time()*1000)
    if track.get("ripe_lock_until", 0) > now_ms and track.get("ripe_state") is not None:
        return track["ripe_state"], float(track["ripe_ema"])

    if track.get("ripe_state") is None:
        track["ripe_state"] = "matang" if track["ripe_ema"] >= 0.5 else "belum matang"
        track["ripe_lock_until"] = now_ms + LOCK_MS
    else:
        if want_state != track["ripe_state"]:
            desired_votes = votes_sum if want_state=="matang" else (len(track["ripe_votes"])-votes_sum)
            if desired_votes >= SWITCH_CONFIRM_FRAMES:
                if want_state=="matang":
                    if (track["ripe_ema"] >= max(RIPE_ON_THRESH, 0.84)) and not green_veto:
                        track["ripe_state"] = "matang"; track["ripe_lock_until"] = now_ms + LOCK_MS
                else:
                    if (track["ripe_ema"] <= RIPE_OFF_THRESH) or green_veto:
                        track["ripe_state"] = "belum matang"; track["ripe_lock_until"] = now_ms + LOCK_MS
    return track["ripe_state"], float(track["ripe_ema"])

# ===== Fallback mask untuk objek gelap di kertas putih =====
def dark_on_white_mask(hsv):
    v = hsv[:,:,2]
    _, m = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K5, 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K7, 2)
    return m

# ============== Tracker per-track =================
_TRACKS = []      # {id,x,y,w,h,score,label,last_seen,misses,created,size_ema,size_state,ripe_ema,ripe_state,ripe_votes,ripe_lock_until,size_last_major,area_ratio_last}
ANALYSIS_MODE = False
_NEXT_TID = 1

def update_tracks(detected: List[dict], frame_hw: Tuple[int,int]) -> List[dict]:
    """
    detected: {x,y,w,h,score,major_px,area_ratio,color_info,probs,top_label}
    Matching pakai IoU + guard (center distance & lonjakan ukuran/area/EMA) → paksa NEW TRACK jika pindah objek.
    Output: NMS + (opsional) ONE_BOX_MODE satu kotak saja dengan prioritas recency.
    """
    global _TRACKS, _NEXT_TID
    now = int(time.time() * 1000)
    H, W = frame_hw
    min_side = float(min(H, W))

    for t in _TRACKS:
        t["matched"] = False

    # 1) Greedy matching + guard
    for det in detected:
        dbox = (det["x"], det["y"], det["w"], det["h"])
        d_cx, d_cy = box_center_xy(dbox)
        best_i, best_t = 0.0, None
        for t in _TRACKS:
            tbox = (t["x"], t["y"], t["w"], t["h"])
            v = iou(dbox, tbox)
            if v > best_i:
                best_i, best_t = v, t

        make_new = False
        if best_t is not None and best_i >= TRACK_MATCH_IOU:
            # Guard checks
            tbox = (best_t["x"], best_t["y"], best_t["w"], best_t["h"])
            t_cx, t_cy = box_center_xy(tbox)
            center_dist_ratio = np.hypot(d_cx - t_cx, d_cy - t_cy) / (min_side + 1e-6)

            prev_major = best_t.get("size_last_major", det["major_px"])
            major_jump_sym = abs(det["major_px"] - prev_major) / (max(1.0, min(det["major_px"], prev_major)))
            major_jump_up  = max(0.0, (det["major_px"] - prev_major)) / (max(1.0, prev_major))

            area_prev  = best_t.get("area_ratio_last", det["area_ratio"])
            area_jump  = abs(det["area_ratio"] - area_prev)

            ema_target = size_metric_from(det["major_px"], det["area_ratio"], frame_hw)
            size_ema   = best_t.get("size_ema", ema_target)
            ema_jump_up = (ema_target - size_ema) / (max(1e-6, size_ema))

            if (center_dist_ratio > CENTER_DIST_RATIO_MAX) or \
               (major_jump_sym > SIZE_RESET_RATIO) or \
               (major_jump_up  > SIZE_RESET_UP) or \
               (ema_jump_up    > SIZE_EMA_RESET_UP) or \
               (area_jump  > AREA_RATIO_RESET_ABS) or \
               (best_i < STRICT_SWITCH_IOU_MAX and (major_jump_up > 0.18 or area_jump > 0.05 or center_dist_ratio > 0.18)):
                make_new = True

        if best_t is None or make_new:
            # NEW TRACK
            tnew = {
                "id": _NEXT_TID, "x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"],
                "score": det.get("score", 1.0), "created": now, "last_seen": now,
                "misses": 0, "matched": True,
                "size_ema": None, "size_state": None,
                "ripe_ema": None, "ripe_state": None,
                "ripe_votes": deque(maxlen=8), "ripe_lock_until": 0,
                "size_last_major": det["major_px"],
                "area_ratio_last": det["area_ratio"]
            }
            size_state, _ = decide_size_track(tnew, det["major_px"], det["area_ratio"], frame_hw)
            ripe_state, _ = fused_ripeness_track(tnew, det.get("probs"), det.get("color_info", {}), det.get("top_label",""))
            tnew["label"] = format_label(ripe_state, size_state)
            save_daily_ripeness(tnew)
            _TRACKS.append(tnew); _NEXT_TID += 1
        else:
            # UPDATE track lama
            best_t.update({
                "x": det["x"], "y": det["y"], "w": det["w"], "h": det["h"],
                "score": max(best_t.get("score", 0.0), det.get("score", 0.0)),
                "last_seen": now, "misses": 0, "matched": True
            })
            size_state, _ = decide_size_track(best_t, det["major_px"], det["area_ratio"], frame_hw)
            ripe_state, _ = fused_ripeness_track(best_t, det.get("probs"), det.get("color_info", {}), det.get("top_label",""))
            best_t["label"] = format_label(ripe_state, size_state)
            save_daily_ripeness(best_t)
            best_t["size_last_major"] = det["major_px"]
            best_t["area_ratio_last"] = det["area_ratio"]

    # 2) Hold & pruning
    out_boxes, new_tracks = [], []
    for t in _TRACKS:
        if not t["matched"]:
            t["misses"] += 1

        since_seen = now - t["last_seen"]
        since_created = now - t["created"]

        hold_ms = TRACK_HOLD_MS
        if t.get("size_state") == "kecil":
            hold_ms += HOLD_BONUS_SMALL_MS

        min_visible_ok = (since_created <= TRACK_MIN_VISIBLE_MS)

        if t["matched"] or since_seen <= hold_ms or min_visible_ok:
            out_boxes.append({
                "x": int(t["x"]), "y": int(t["y"]), "w": int(t["w"]), "h": int(t["h"]),
                "label": t.get("label", ""), "score": float(t.get("score", 1.0)),
                "tid": t["id"],
                "last_seen": t["last_seen"],
                "created": t["created"]
            })

        if (since_seen <= TRACK_DROP_MS) and (t["misses"] <= TRACK_MAX_MISSES):
            new_tracks.append(t)
    _TRACKS = new_tracks

    if out_boxes:
        out_boxes = nms_boxes(out_boxes, NMS_IOU_THRESH)
        if ONE_BOX_MODE:
            def pri(b):
                area = b["w"] * b["h"]
                return (b["last_seen"], b["score"], area)
            out_boxes.sort(key=pri, reverse=True)
            out_boxes = out_boxes[:1]

    if not ONE_BOX_MODE and len(out_boxes) > TOPK_BOXES:
        out_boxes = out_boxes[:TOPK_BOXES]
    return out_boxes

# ===================== Deteksi per frame =================
def detect_avocado(img_bgr, flash_on=False):
    t0 = time.time()
    H0, W0 = img_bgr.shape[:2]
    if W0 > MAX_W:
        scale = MAX_W / float(W0)
        img_bgr = cv2.resize(img_bgr, (int(W0 * scale), int(H0 * scale)))

    img_bgr = gray_world(img_bgr)
    blur    = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    hsv     = maybe_clahe_hsv(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV))

    red, yellow, green, olive, dark = color_masks(hsv)
    mask_all = red | yellow | green | olive | dark

    cnts, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 700]

    H, W = hsv.shape[:2]
    detects = []
    neg_candidates = []  # kandidat negatif
    color_info_dbg = None

    def process_cnt(c):
        nonlocal color_info_dbg
        ok_shape, _ = avocado_shape_rule(c, (H, W))
        if not ok_shape: 
            return

        # Cek veto non-alpukat lebih dulu
        non_av, neg_score, _reasons = is_non_avocado(img_bgr, hsv, c, (H, W), flash_on)
        if non_av:
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) > NEG_MIN_AREA and NEGATIVE_MODE:
                neg_candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": max(0.60, neg_score)})
            return

        # Lolos veto → cek warna signature alpukat
        blob = np.zeros((H, W), np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        ok_color, color_info = avocado_color_rule(hsv, blob, (red, yellow, green, olive, dark))
        if not ok_color:
            if NEGATIVE_MODE and cv2.contourArea(c) > NEG_MIN_AREA:
                x, y, w, h = cv2.boundingRect(c)
                neg_candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": 0.70})
            return
        
        # === HARD STOP: terlalu gelap → bukan alpukat ===
        if color_info:
            if color_info.get("dark", 0) >= 0.75 and color_info.get("green", 0) < 0.12:
                return

        # ---- jalur positif (alpukat) + SVM (opsional untuk speed) ----
        probs = None
        top_label = ""
        top_conf = 1.0

        if USE_SVM:
            feat = feature_vector(hsv, blob, c, (red, yellow, green, olive, dark)).reshape(1, -1)
            if hasattr(PIPELINE, "predict_proba"):
                probs = PIPELINE.predict_proba(feat)[0]
                idx   = int(np.argmax(probs)); top_label = label_from_idx(idx)
                top_conf  = float(probs[idx])
                if top_conf < MIN_PROBA:
                    return
            else:
                top_label = label_from_pred_value(PIPELINE.predict(feat)[0])
                top_conf = 1.0
        else:
            # skip svm: cukup lolos rule warna+shape, biar cepat
            top_label = "rule"
            top_conf = 1.0


        x, y, w, h = cv2.boundingRect(c)
        major = float(max(w, h))
        if USE_ELLIPSE and len(c) >= 5:
            try:
                (_, _), (MA, ma), _ = cv2.fitEllipse(c)
                major = float(max(MA, ma))
            except Exception:
                pass
        area_ratio_contour = cv2.contourArea(c) / float(H*W)

        detects.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "score": float(top_conf),
            "major_px": major,
            "area_ratio": area_ratio_contour,
            "color_info": color_info or {},
            "probs": probs,
            "top_label": top_label
        })
        color_info_dbg = color_info

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        process_cnt(c)
        if len(detects) >= TOPK_BOXES:
            break

    if not detects and FALLBACK_DARK_ON_WHITE:
        m_fb = dark_on_white_mask(hsv)
        cnts2, _ = cv2.findContours(m_fb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts2, key=cv2.contourArea, reverse=True):
            ok_shape, _ = avocado_shape_rule(c, (H, W))
            if not ok_shape:
                continue
            non_av, neg_score, _reasons = is_non_avocado(img_bgr, hsv, c, (H, W), flash_on)
            if non_av:
                if NEGATIVE_MODE and cv2.contourArea(c) > NEG_MIN_AREA:
                    x, y, w, h = cv2.boundingRect(c)
                    neg_candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": max(0.60, neg_score)})
                continue

            blob = np.zeros((H, W), np.uint8); cv2.drawContours(blob, [c], -1, 255, -1)
            
            probs = None
            top_conf = 1.0
            top_label = ""

            if USE_SVM:
                feat = feature_vector(hsv, blob, c, (red, yellow, green, olive, dark)).reshape(1, -1)
                if hasattr(PIPELINE, "predict_proba"):
                    probs = PIPELINE.predict_proba(feat)[0]
                    idx   = int(np.argmax(probs)); top_conf  = float(probs[idx])
                    if top_conf < 0.50: 
                        if NEGATIVE_MODE and cv2.contourArea(c) > NEG_MIN_AREA:
                            x, y, w, h = cv2.boundingRect(c)
                            neg_candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": 0.65})
                        continue
                    top_label = label_from_idx(idx)
                else:
                    top_label = label_from_pred_value(PIPELINE.predict(feat)[0])
                    top_conf = 1.0
            else:
                top_label = "rule"
                top_conf = 1.0

            x, y, w, h = cv2.boundingRect(c)
            major = float(max(w, h))
            if USE_ELLIPSE and len(c) >= 5:
                try:
                    (_, _), (MA, ma), _ = cv2.fitEllipse(c)
                    major = float(max(MA, ma))
                except Exception:
                    pass
            area_ratio_contour = cv2.contourArea(c) / float(H*W)
            detects.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "score": float(top_conf),
                "major_px": major,
                "area_ratio": area_ratio_contour,
                "color_info": {},  # fallback gelap
                "probs": probs,
                "top_label": top_label
            })
            if len(detects) >= TOPK_BOXES:
                break

    boxes = update_tracks(detects, (H, W))

    # === Fallback negatif: jika tidak ada alpukat tetapi ada kandidat non-alpukat, tampilkan BUKAN ALPUKAT ===
    if not boxes and NEGATIVE_MODE and neg_candidates:
        neg_candidates.sort(key=lambda b: (b["score"], b["w"]*b["h"]), reverse=True)
        b = neg_candidates[0]
        boxes = [{
            "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
            "label": NEG_LABEL, "score": float(b["score"]),
            "tid": -1, "last_seen": int(time.time()*1000), "created": int(time.time()*1000)
        }]

    t_ms = (time.time() - t0) * 1000.0
    final_label = boxes[0]["label"] if boxes else ""
    final_conf  = boxes[0]["score"] if boxes else 0.0
    vis = None
    if RETURN_MODE == "frame":
        vis = img_bgr.copy()
        if not boxes:
            put_text_with_bg(vis, "Tidak ada alpukat terdeteksi", 12, 32, scale=0.7)
        else:
            for bx in boxes:
                draw_box_with_label(vis, (bx["x"], bx["y"], bx["w"], bx["h"]), bx["label"], bx["score"])
            if color_info_dbg:
                ci = color_info_dbg
                dbg = f"g:{ci.get('green',0):.2f} ol:{ci.get('olive',0):.2f} dr:{ci.get('dark',0):.2f} r:{ci.get('red',0):.2f} y:{ci.get('yellow',0):.2f}"
                put_text_with_bg(vis, dbg, 12, H-12, scale=0.55, bg_color=(20,20,20))
    return vis, final_label, final_conf, t_ms, boxes, (W, H)

# ===================== Server (Flask) + Auth via DB =================
app = Flask(__name__, static_folder="static", template_folder="templates")

app.secret_key = os.getenv("SECRET_KEY", "dev_secret_change_me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL","mysql+pymysql://root:@localhost/alpukat_db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = "users"
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# ===================== Riwayat Kematangan Harian =====================
class AvocadoHistory(db.Model):
    __tablename__ = "avocado_history"

    id = db.Column(db.Integer, primary_key=True)
    track_id = db.Column(db.Integer, nullable=False)
    size = db.Column(db.String(10), nullable=False)
    ripe_score = db.Column(db.Float, nullable=False)
    ripe_label = db.Column(db.String(20), nullable=False)
    date = db.Column(db.Date, nullable=False)
    
with app.app_context():
    db.create_all()


@app.before_request
def _auth_gate():
    if request.endpoint == "login_form" and session.get("user"):
        return redirect(url_for("dashboard"))

@app.get("/")
def landing():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return render_template("landing.html")

@app.get("/register")
def register_form():
    if request.args.get("force") == "1":
        session.pop("user", None); session.pop("uid", None)
        return render_template("register.html", error="", success="")
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return render_template("register.html", error="", success="")

@app.post("/register")
def register_submit():
    username = (request.form.get("username") or "").strip()
    password = (request.form.get("password") or "").strip()
    confirm  = (request.form.get("confirm")  or "").strip()
    if not username or not password:
        flash("Username dan password wajib diisi.", "error")
        return redirect(url_for("register_form", force=1))
    if len(password) < 6:
        flash("Password minimal 6 karakter.", "error")
        return redirect(url_for("register_form", force=1))
    if password != confirm:
        flash("Konfirmasi password tidak cocok.", "error")
        return redirect(url_for("register_form", force=1))
    try:
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user); db.session.commit()
    except IntegrityError:
        db.session.rollback()
        flash("Username sudah digunakan.", "error")
        return redirect(url_for("register_form", force=1))
    flash("Akun berhasil dibuat. Silakan login.", "success")
    return redirect(url_for("login_form", force=1))

@app.get("/login")
def login_form():
    if request.args.get("force") == "1":
        session.pop("user", None); session.pop("uid", None)
        success = session.pop("flash_success", "")
        return render_template("login.html", error="", success=success)
    if session.get("user"):
        return redirect(url_for("dashboard"))
    success = session.pop("flash_success", "")
    return render_template("login.html", error="", success=success)

@app.post("/login")
def login_submit():
    u = (request.form.get("username") or "").strip()
    p = (request.form.get("password") or "").strip()
    user = User.query.filter_by(username=u).first()
    if not user or not check_password_hash(user.password, p):
        flash("Username atau password salah.", "error")
        return redirect(url_for("login_form"))
    session["user"] = user.username; session["uid"] = user.id
    flash(f"Login berhasil. Selamat datang, {user.username}!", "success")
    return redirect(url_for("dashboard"))

@app.get("/logout")
def logout():
    session.clear()
    flash("Anda berhasil logout.", "success")
    return redirect(url_for("login_form"))

@app.get("/dashboard")
def dashboard():
    if not session.get("user"):
        return redirect(url_for("login_form"))
    get_flashed_messages(with_categories=True)
    return render_template("index.html")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/modelinfo")
def modelinfo():
    model_classes = list(map(str, CLF_CLASSES)) if CLF_CLASSES is not None else None
    return {"ok": True, "model": "SVM (RBF)", "classes": model_classes or CLASSES}

@app.post("/infer")
def infer():
    flash_on = request.headers.get("X-Flash-On", "0") == "1"
    if not session.get("user"):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    try:
        raw = request.data
        if not raw:
            return jsonify({"ok": False, "error": "empty body"}), 400
        npbuf = np.frombuffer(raw, np.uint8)
        img   = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"ok": False, "error": "decode fail"}), 400
        vis, label, conf, t_ms, boxes, (W, H) = detect_avocado(img, flash_on)
        if RETURN_MODE == "boxes":
            return jsonify({"ok": True, "boxes": boxes, "label": label, "score": conf,
                            "t_ms": t_ms, "input_w": W, "input_h": H})
        else:
            ok, buf = cv2.imencode(".jpg", vis if vis is not None else img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                return jsonify({"ok": False, "error": "encode fail"}), 500
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return jsonify({"ok": True, "frame": b64, "label": label, "score": conf,
                            "t_ms": t_ms, "input_w": W, "input_h": H})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    
@app.get("/api/avocado-history")
def avocado_history_chart():
    rows = (
        db.session.query(
            AvocadoHistory.date,
            AvocadoHistory.size,
            db.func.avg(AvocadoHistory.ripe_score).label("avg_score")
        )
        .group_by(
            AvocadoHistory.date,
            AvocadoHistory.size
        )
        .order_by(AvocadoHistory.date)
        .all()
    )

    dates = sorted({r.date for r in rows})

    data = {
        "labels": [d.strftime("%d-%m") for d in dates],
        "kecil": [],
        "besar": []
    }

    for d in dates:
        kecil = next((r.avg_score for r in rows if r.date == d and r.size == "kecil"), None)
        besar = next((r.avg_score for r in rows if r.date == d and r.size == "besar"), None)

        data["kecil"].append(kecil)
        data["besar"].append(besar)

    return {"ok": True, **data}

@app.post("/toggle-analysis")
def toggle_analysis():
    global ANALYSIS_MODE
    ANALYSIS_MODE = not ANALYSIS_MODE
    return {"ok": True, "analysis_mode": ANALYSIS_MODE}


# ===================== MAIN =====================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    print(f"[INFO] Model: {MODEL_PATH}")
    print(f"[INFO] RETURN_MODE: {RETURN_MODE} (boxes lebih cepat)")
    print(f"[RUN] http://{host}:{port}")
    app.run(host=host, port=port, debug=False)