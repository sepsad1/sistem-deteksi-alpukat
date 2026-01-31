# train_svm.py — SVM RBF + CM hold-out 20% & OOF
# + export normalized RGB/HSV
# + CSV: area & perimeter DISKALAKAN (area ~ ribuan seperti perimeter)
# + dataset splitter 80:20 (MOVE by default with --split)
# + tulis predicted label & confidence utk 20% hold-out

import os, glob, sys, joblib, csv, shutil
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional

from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split, cross_val_predict
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
)
import matplotlib.pyplot as plt

# ====================== Kelas ======================
CLASSES = ["matang_besar","matang_kecil","belummatang_besar","belummatang_kecil"]

# ====================== Skala CSV ======================
# area_csv  = AREA_K  * (area_ratio  ** AREA_GAMMA),  area_ratio=area/(H*W)
# peri_csv  = PERIM_K * (perim_ratio ** PERIM_GAMMA), perim_ratio=perimeter/(2*(H+W))
# Default baru: area jadi ~RIBUAN (selevel perimeter).
AREA_GAMMA  = float(os.getenv("AREA_GAMMA",  "2.0"))
AREA_K      = float(os.getenv("AREA_K",      "400000.0"))  # NAIK agar area ~ ribuan
PERIM_GAMMA = float(os.getenv("PERIM_GAMMA", "2.27"))
PERIM_K     = float(os.getenv("PERIM_K",     "33126.0"))

# ============ HSV masking & avocado gate ============
def build_mask(hsv, lower, upper):
    mask = cv2.inRange(hsv, np.array(lower,np.uint8), np.array(upper,np.uint8))
    mask = cv2.medianBlur(mask, 5)
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
    return mask

def color_masks(hsv):
    red1 = build_mask(hsv, (0,100,40), (10,255,255))
    red2 = build_mask(hsv, (170,100,40), (180,255,255))
    red  = cv2.bitwise_or(red1, red2)
    yellow = build_mask(hsv, (10,35,25), (38,255,255))
    green  = build_mask(hsv, (35,25,20), (90,255,255))
    olive  = build_mask(hsv, (20,25,20), (45,200,150))
    dark   = build_mask(hsv, (0,0,0), (180,140,110))
    return red, yellow, green, olive, dark

def avocado_gate(hsv, contour, masks, min_area_ratio=0.003):
    H,W = hsv.shape[:2]
    area = cv2.contourArea(contour)
    if area < 800 or (area/(H*W)) < min_area_ratio:
        return False
    blob = np.zeros((H,W), np.uint8)
    cv2.drawContours(blob,[contour],-1,255,-1)
    total = cv2.countNonZero(blob)
    if total == 0: return False

    green, yellow, olive, dark = masks
    def ratio(m): return cv2.countNonZero(cv2.bitwise_and(m, blob))/float(total)
    r_g, r_y, r_o, r_d = ratio(green), ratio(yellow), ratio(olive), ratio(dark)
    color_score = max(r_g + 0.6*r_y + 0.4*r_o, r_d*0.85 + 0.2*r_g)
    if color_score < 0.28: return False

    hull = cv2.convexHull(contour); hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area>0 else 0.0
    perim = cv2.arcLength(contour, True)
    circularity = 4*np.pi*area/(perim*perim) if perim>0 else 0.0
    x,y,w,h = cv2.boundingRect(contour)
    ar = w/float(h) if h>0 else 1.0
    if not (solidity>=0.78 and 0.28<=circularity<=0.98 and 0.40<=ar<=2.70):
        return False
    return True

# =================== Fitur (HARUS identik dgn app.py) ====================
FEAT_NAMES = (
    [f"hbin_{i}" for i in range(16)] +
    ["s_mean","v_mean"] +
    ["ratio_red","ratio_yellow","ratio_green","ratio_olive","ratio_dark"] +
    ["area_ratio","bbox_ratio","major_axis_ratio"] +
    ["solidity","circularity","aspect_ratio"]
)

def feature_vector(hsv, blob_mask, cnt, masks) -> np.ndarray:
    H,W = hsv.shape[:2]
    hist = cv2.calcHist([hsv],[0],blob_mask,[16],[0,180]).astype(np.float32).flatten()
    hist = hist/(hist.sum()+1e-8)
    s_mean = cv2.mean(hsv[:,:,1], mask=blob_mask)[0]/255.0
    v_mean = cv2.mean(hsv[:,:,2], mask=blob_mask)[0]/255.0

    total = cv2.countNonZero(blob_mask)
    rs = []
    for m in masks:
        rs.append(cv2.countNonZero(cv2.bitwise_and(m, blob_mask))/float(total) if total>0 else 0.0)

    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    area_ratio = area/float(H*W)
    bbox_ratio = (w*h)/float(H*W)
    major = max(w,h)
    if len(cnt) >= 5:
        try:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt); major = max(MA, ma)
        except:
            pass
    major_axis_ratio = major/float(min(H,W))

    hull = cv2.convexHull(cnt); hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area>0 else 0.0
    perim = cv2.arcLength(cnt, True)
    circularity = 4*np.pi*area/(perim*perim) if perim>0 else 0.0
    aspect_ratio = w/float(h) if h>0 else 1.0

    return np.concatenate([
        hist,
        np.array([s_mean, v_mean], np.float32),
        np.array(rs, np.float32),
        np.array([area_ratio, bbox_ratio, major_axis_ratio,
                  solidity, circularity, aspect_ratio], np.float32)
    ])

# ============ Ekstraksi 1 gambar + METRICS (normalized only) ============
def extract_from_image(path: str) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    blur = cv2.GaussianBlur(img,(5,5),0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    red, yellow, green, olive, dark = color_masks(hsv)
    mask_all = red | yellow | green | olive | dark

    cnts, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 600]
    if not cnts:
        return None, None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        if avocado_gate(hsv, c, (green,yellow,olive,dark)):
            blob = np.zeros(mask_all.shape, np.uint8)
            cv2.drawContours(blob,[c],-1,255,-1)

            feat = feature_vector(hsv, blob, c, (red,yellow,green,olive,dark))

            # Mean warna (raw)
            b_mean, g_mean, r_mean, _ = cv2.mean(blur, mask=blob)
            h_mean, s_mean, v_mean, _ = cv2.mean(hsv,  mask=blob)

            # Normalisasi warna 0..1
            Rn = float(r_mean) / 255.0
            Gn = float(g_mean) / 255.0
            Bn = float(b_mean) / 255.0
            Hn = (float(h_mean) * 2.0) / 360.0
            Sn = float(s_mean) / 255.0
            Vn = float(v_mean) / 255.0

            # Area & perimeter asli (px^2 dan px)
            area_px_raw      = float(cv2.contourArea(c))
            perimeter_px_raw = float(cv2.arcLength(c, True))

            # ----- Skala agar area ~ RIBUAN (selevel perimeter) -----
            H, W = hsv.shape[:2]
            area_ratio  = area_px_raw / float(H * W)               # 0..1
            perim_ratio = perimeter_px_raw / float(2 * (H + W))    # 0..1
            area_csv = AREA_K  * (area_ratio  ** AREA_GAMMA)
            peri_csv = PERIM_K * (perim_ratio ** PERIM_GAMMA)

            metrics = {
                "path": path,
                "label": "",  # diisi saat load_dataset
                "R_norm": round(Rn,4),
                "G_norm": round(Gn,4),
                "B_norm": round(Bn,4),
                "H_norm": round(Hn,4),
                "S_norm": round(Sn,4),
                "V_norm": round(Vn,4),
                "area_px": round(float(area_csv), 3),
                "perimeter_px": round(float(peri_csv), 3),
            }
            return feat, metrics
    return None, None

# =================== Split dataset 80:20 (MOVE) ====================
def split_dataset_80_20(root_dir: str,
                        out_root: str = "dataset_split",
                        test_size: float = 0.20,
                        seed: int = 42,
                        copy: bool = False):
    rng = np.random.RandomState(seed)
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")

    items = []  # (path, label)
    for cls in CLASSES:
        folder = Path(root_dir)/cls
        files = []
        for ext in exts:
            files += glob.glob(str(folder/ext))
        for f in files:
            items.append((f, cls))

    if not items:
        raise RuntimeError(f"Tidak ada file di {root_dir}")

    split_map = {"train": [], "test": []}
    for cls in CLASSES:
        cls_files = [p for p,l in items if l==cls]
        n = len(cls_files)
        if n == 0:
            continue
        idx = np.arange(n); rng.shuffle(idx)
        n_test = max(1, int(round(test_size * n)))
        test_idx = set(idx[:n_test].tolist())
        for i, fp in enumerate(cls_files):
            split = "test" if i in test_idx else "train"
            split_map[split].append((fp, cls))

    for split in ("train","test"):
        for _, cls in split_map[split]:
            Path(out_root, split, cls).mkdir(parents=True, exist_ok=True)

    def _place(src, dst):
        if copy: shutil.copy2(src, dst)
        else:    shutil.move(src, dst)

    for split in ("train","test"):
        for src, cls in split_map[split]:
            dst = Path(out_root, split, cls, Path(src).name)
            if not dst.exists():
                _place(src, str(dst))

    for split in ("train","test"):
        csv_path = Path(out_root, f"{split}_files.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["path","label"])
            for p, cls in split_map[split]:
                w.writerow([str(Path(out_root, split, cls, Path(p).name)), cls])

    print(f"[SPLIT] Selesai (mode {'COPY' if copy else 'MOVE'}). Train/Test di: {out_root}")
    for split in ("train","test"):
        by_cls = {c:0 for c in CLASSES}
        for _, c in split_map[split]: by_cls[c]+=1
        print(f"        {split}: " + ", ".join(f"{c}:{by_cls[c]}" for c in CLASSES))

    return str(Path(out_root,"train")), str(Path(out_root,"test"))

# =================== Loader dataset (fitur + metrics) ====================
def load_dataset(root: str):
    X, y, metrics_list = [], [], []
    miss, total = 0, 0
    per_cls_count = {c:0 for c in CLASSES}

    for cls in CLASSES:
        folder = Path(root)/cls
        files = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            files += glob.glob(str(folder/ext))
        for f in files:
            total += 1
            feat, metrics = extract_from_image(f)
            if feat is None:
                miss += 1
                continue
            X.append(feat); y.append(cls)
            metrics["label"] = cls
            metrics_list.append(metrics)
            per_cls_count[cls] += 1

    if len(X) == 0:
        raise RuntimeError("Tidak ada fitur yang berhasil diekstrak. Cek gate & dataset.")

    print(f"[TRAIN] Baca dataset: {len(X)} sampel dari {total} file (skip {miss})")
    for cls in CLASSES:
        print(f"        - {cls:22s}: {per_cls_count[cls]}")

    return np.array(X, np.float32), np.array(y), metrics_list

# =================== Simpan metrics CSV (normalized only) ====================
def save_metrics_csv(metrics_list: List[dict], out_csv: str):
    if not metrics_list:
        print("[WARN] Tidak ada metrics untuk disimpan."); return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    keys = [
        "path","label",
        "R_norm","G_norm","B_norm","H_norm","S_norm","V_norm",
        "area_px","perimeter_px",
        "pred_label","pred_conf"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for m in metrics_list:
            w.writerow({k: m.get(k, "") for k in keys})
    print("[OK] Simpan metrics:", out_csv)

# =================== Plot & simpan CM helper ===================
def save_cm(cm: np.ndarray, labels: List[str], path: str, title: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig = disp.plot(cmap="Blues", xticks_rotation=45).figure_
    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print("[OK] Simpan:", path)

# =================== Train + CM hold-out 20% & OOF ===================
def main():
    data_dir = sys.argv[1] if len(sys.argv)>1 else "dataset"
    use_split = ("--split" in sys.argv) or ("-s" in sys.argv)

    out_path = "models/avocado_svm.joblib"
    cm_holdout_path = "models/confusion_matrix_holdout_20.png"
    cm_oof_path     = "models/confusion_matrix_oof.png"
    metrics_csv_tr  = "models/train_metrics.csv"
    metrics_csv_te  = "models/test_metrics.csv"
    os.makedirs("models", exist_ok=True)

    if use_split:
        train_dir, test_dir = split_dataset_80_20(
            data_dir, out_root="dataset_split", test_size=0.20, seed=42, copy=False
        )
        X_tr, y_tr, metrics_tr = load_dataset(train_dir)
        X_te, y_te, metrics_te = load_dataset(test_dir)
    else:
        X_all, y_all, metrics_all = load_dataset(data_dir)
        idx_all = np.arange(len(X_all))
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            X_all, y_all, idx_all, test_size=0.20, stratify=y_all, random_state=42
        )
        metrics_tr = [metrics_all[i] for i in idx_tr]
        metrics_te = [metrics_all[i] for i in idx_te]

    save_metrics_csv(metrics_tr, metrics_csv_tr)

    pipe = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", probability=True)
    )
    param_grid = {"svc__C":[1,3,10,30,100], "svc__gamma":["scale",0.1,0.03,0.01,0.003]}
    cv_gs = StratifiedKFold(n_splits=min(5, len(y_tr)//4 if len(y_tr)//4>=2 else 2),
                            shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv_gs, n_jobs=-1, verbose=1)
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    print("[TRAIN] Best params:", gs.best_params_, "| CV acc:", gs.best_score_)

    y_pred = best.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"[TEST] Hold-out(20%) Accuracy: {acc:.3f}")
    cm_hold = confusion_matrix(y_te, y_pred, labels=CLASSES)
    save_cm(cm_hold, CLASSES, cm_holdout_path, "Confusion Matrix - Hold-out 20%")
    print("\n[TEST] Hold-out(20%) Classification report:\n",
          classification_report(y_te, y_pred, labels=CLASSES, zero_division=0))

    if hasattr(best, "predict_proba"):
        prob = best.predict_proba(X_te)
        pred_idx = np.argmax(prob, axis=1)
        pred_conf = [float(prob[i, j]) for i, j in enumerate(pred_idx)]
    else:
        pred_conf = ["" for _ in range(len(y_pred))]

    for i in range(len(metrics_te)):
        metrics_te[i]["pred_label"] = str(y_pred[i])
        if pred_conf:
            metrics_te[i]["pred_conf"] = round(float(pred_conf[i]), 4)

    save_metrics_csv(metrics_te, metrics_csv_te)

    best_C = best.get_params()["svc__C"]; best_gamma = best.get_params()["svc__gamma"]
    base = make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=best_C, gamma=best_gamma,
                             class_weight="balanced", probability=False))
    if not use_split:
        X_all = np.concatenate([X_tr, X_te], axis=0)
        y_all = np.concatenate([y_tr, y_te], axis=0)
    cv_oof = StratifiedKFold(n_splits=min(5, len(y_tr)//4 if len(y_tr)//4>=2 else 2),
                             shuffle=True, random_state=42)
    y_oof = cross_val_predict(base, X_tr, y_tr, cv=cv_oof, n_jobs=-1)
    cm_oof = confusion_matrix(y_tr, y_oof, labels=CLASSES)
    save_cm(cm_oof, CLASSES, cm_oof_path, "Confusion Matrix - OOF (Train Only)")

    if use_split:
        X_all = np.concatenate([X_tr, X_te], axis=0)
        y_all = np.concatenate([y_tr, y_te], axis=0)
    best.fit(X_all, y_all)
    artifact = {"pipeline": best, "classes": CLASSES, "feature_names": FEAT_NAMES}
    joblib.dump(artifact, out_path)
    print("[OK] Model tersimpan:", out_path)
    print("[OK] CSV dengan prediksi test:", metrics_csv_te)

if __name__ == "__main__":
    main()
