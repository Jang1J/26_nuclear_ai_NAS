"""Late Correction 빠른 테스트 — 5ep만, 핵심 파라미터만."""
import os, sys, csv, warnings
from pathlib import Path
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

NAS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NAS_DIR / "Team N code" / "py"))

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]
LOCA_INDICES = {1, 2, 3}

MODEL_DIR = NAS_DIR / "models_9class_v3_ss5"
DATA_ROOT = NAS_DIR / "Team N code" / "data"
ANSWERS_PATH = NAS_DIR / "data" / "real_test_data" / "answers.csv"


def sanitize_array(arr):
    return np.where(np.isfinite(arr), arr, 0.0)


def run_all_tests(model, scaler, feat, raw_names):
    """100건 모두 추론하고, 각 sec별 예측 히스토리를 저장."""
    answers = {}
    with open(ANSWERS_PATH, "r") as f:
        for row in csv.DictReader(f):
            answers[int(row["test_id"])] = row["label"]

    results = {}  # tid -> {preds, probs, confirmed_label, confirmed_idx, confirm_sec}

    for tid in range(1, 101):
        folder = DATA_ROOT / f"test{tid}"
        if not folder.exists():
            continue

        sec_files = [p for p in folder.glob(f"test{tid}_sec*.csv") if " " not in p.name]
        sec_files = sorted(sec_files, key=lambda p: int(p.stem.split("sec")[1]))
        if not sec_files:
            continue

        buffer = []
        preds = []
        probs = []
        confirmed = False
        confirmed_label = None
        confirmed_idx = None
        confirm_sec = None
        header = None
        _cached_header = None
        _cached_col_to_idx = None

        for sf in sec_files:
            sec = int(sf.stem.split("sec")[1])
            with open(sf, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                continue
            col_names = rows[0]
            data_row = rows[1]
            if header is None:
                header = col_names

            if _cached_header != col_names:
                _cached_col_to_idx = {name: i for i, name in enumerate(col_names)}
                _cached_header = col_names

            values = []
            for v in data_row:
                try:
                    val = float(v)
                    if not np.isfinite(val): val = 0.0
                except: val = 0.0
                values.append(val)
            x_raw = np.array(values, dtype=np.float32)
            x_raw = sanitize_array(x_raw)

            x_selected = np.array(
                [x_raw[_cached_col_to_idx[name]] if name in _cached_col_to_idx else 0.0
                 for name in raw_names], dtype=np.float32)
            x_selected = sanitize_array(x_selected)
            buffer.append(x_selected)

            X_buf = np.array(buffer, dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_feat, _, _ = feat.transform(X_buf, np.zeros(len(X_buf)))
            X_feat = sanitize_array(X_feat)
            X_scaled = scaler.transform(X_feat)
            X_scaled = sanitize_array(X_scaled)
            X_scaled = np.clip(X_scaled, -10, 10)

            t = X_scaled.shape[0]
            WINDOW = 3
            if t < WINDOW:
                pad = np.tile(X_scaled[0:1], (WINDOW - t, 1))
                window = np.vstack([pad, X_scaled])
            else:
                window = X_scaled[t - WINDOW: t]

            prob = model.predict(window[np.newaxis, ...], verbose=0)[0]
            if not np.all(np.isfinite(prob)):
                prob = np.ones(len(LABELS)) / len(LABELS)
            pred = int(np.argmax(prob))

            preds.append(pred)
            probs.append(prob.copy())

            if not confirmed and pred != 0 and len(preds) > 5:
                if len(preds) >= 3:
                    recent = preds[-3:]
                    if all(p == pred for p in recent):
                        confirmed = True
                        confirmed_label = LABELS[pred]
                        confirmed_idx = pred
                        confirm_sec = sec

        results[tid] = {
            "preds": preds,
            "probs": probs,
            "confirmed_label": confirmed_label or "NORMAL",
            "confirmed_idx": confirmed_idx if confirmed_idx is not None else 0,
            "confirm_sec": confirm_sec,
            "ans": answers.get(tid, "?"),
        }

    return results


def apply_late_correction(results, late_window, late_conf):
    """사전 계산된 결과에 Late Correction 적용."""
    correct_orig = 0
    correct_late = 0
    corrections = []

    for tid in sorted(results.keys()):
        r = results[tid]
        ans = r["ans"]
        orig_label = r["confirmed_label"]
        orig_idx = r["confirmed_idx"]
        preds = r["preds"]
        probs_list = r["probs"]

        # Late Correction
        final_label = orig_label

        if orig_idx in LOCA_INDICES and len(preds) >= 3:
            recent_preds = preds[-late_window:]
            recent_probs = probs_list[-late_window:]

            loca_preds = [p for p in recent_preds if p in LOCA_INDICES]
            if loca_preds:
                counter = Counter(loca_preds)
                most_idx, most_count = counter.most_common(1)[0]

                if most_idx != orig_idx:
                    ratio = most_count / len(loca_preds)
                    avg_conf = np.mean([p[most_idx] for p in recent_probs])

                    if ratio >= 0.5 and avg_conf >= late_conf:
                        final_label = LABELS[most_idx]

        if orig_label == ans:
            correct_orig += 1
        if final_label == ans:
            correct_late += 1
        if orig_label != final_label:
            ok_o = "O" if orig_label == ans else "X"
            ok_l = "O" if final_label == ans else "X"
            corrections.append((tid, ans, orig_label, final_label, ok_o, ok_l))

    return correct_orig, correct_late, corrections


def main():
    print("=" * 80)
    print("  Late Correction Quick Test")
    print("=" * 80)

    # 5ep 모델
    prefix = "tcn__feat=physics_v3__val=1__ep=5__cw=1__seed=0__skipd=5__dan=1__ss=5__win=3__stride=1"
    model = tf.keras.models.load_model(str(MODEL_DIR / f"{prefix}__model.keras"), compile=False)
    scaler = joblib.load(MODEL_DIR / f"{prefix}__scaler.pkl")
    feat = joblib.load(MODEL_DIR / f"{prefix}__feature_transformer.pkl")
    raw_names = feat.feature_names_all

    dummy = np.zeros((1, 3, len(feat.feature_names)), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    print("\n[5ep] 100건 추론 중...")
    results_5 = run_all_tests(model, scaler, feat, raw_names)
    print(f"[5ep] {len(results_5)}건 완료")

    # 100ep 모델
    prefix100 = "tcn__feat=physics_v3__val=1__ep=100__cw=1__seed=0__skipd=5__dan=1__ss=5__win=3__stride=1"
    model100 = tf.keras.models.load_model(str(MODEL_DIR / f"{prefix100}__model.keras"), compile=False)
    scaler100 = joblib.load(MODEL_DIR / f"{prefix100}__scaler.pkl")
    feat100 = joblib.load(MODEL_DIR / f"{prefix100}__feature_transformer.pkl")
    raw100 = feat100.feature_names_all
    _ = model100.predict(dummy, verbose=0)

    print("\n[100ep] 100건 추론 중...")
    results_100 = run_all_tests(model100, scaler100, feat100, raw100)
    print(f"[100ep] {len(results_100)}건 완료")

    # Late Correction 파라미터 탐색
    print("\n" + "=" * 80)
    print("  5에폭 Late Correction")
    print("=" * 80)

    for lw in [5, 10, 15, 20]:
        for lc in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            orig, late, corr = apply_late_correction(results_5, lw, lc)
            delta = late - orig
            marker = " ***" if delta > 0 else (" !!!" if delta < 0 else "")
            print(f"  lw={lw:>2d} conf={lc:.1f}: {orig}/100 -> {late}/100 (delta={delta:+d}){marker}")
            if corr:
                for tid, ans, o, f, ok_o, ok_l in corr:
                    print(f"    test{tid:>3d}: {ans:>10s}  {o:>10s}({ok_o})->{f:>10s}({ok_l})")

    print("\n" + "=" * 80)
    print("  100에폭 Late Correction")
    print("=" * 80)

    for lw in [5, 10, 15, 20]:
        for lc in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
            orig, late, corr = apply_late_correction(results_100, lw, lc)
            delta = late - orig
            marker = " ***" if delta > 0 else (" !!!" if delta < 0 else "")
            print(f"  lw={lw:>2d} conf={lc:.1f}: {orig}/100 -> {late}/100 (delta={delta:+d}){marker}")
            if corr:
                for tid, ans, o, f, ok_o, ok_l in corr:
                    print(f"    test{tid:>3d}: {ans:>10s}  {o:>10s}({ok_o})->{f:>10s}({ok_l})")


if __name__ == "__main__":
    main()
