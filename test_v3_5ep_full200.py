"""V3 5ep 모델 200건 전체 테스트 + 상세 오답 분석."""
import os, sys, csv, warnings, time
from pathlib import Path
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

NAS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NAS_DIR / "Team N code" / "py"))
from main import load_pipeline, RealtimeInference, LABELS

DATA_ROOT = NAS_DIR / "Team N code" / "data"
ANSWERS_PATH = NAS_DIR / "data" / "real_test_data" / "answers.csv"


def main():
    answers = {}
    with open(ANSWERS_PATH, "r") as f:
        for row in csv.DictReader(f):
            answers[int(row["test_id"])] = row["label"]

    model, scaler, feat, raw_names = load_pipeline()
    inf = RealtimeInference(model, scaler, feat, raw_names)
    dummy = np.zeros((1, inf.WINDOW, len(feat.feature_names)), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    correct = 0
    errors = []
    total = 0

    for tid in range(1, 201):
        folder = DATA_ROOT / f"test{tid}"
        if not folder.exists():
            continue
        total += 1

        sec_files = [p for p in folder.glob(f"test{tid}_sec*.csv") if " " not in p.name]
        sec_files = sorted(sec_files, key=lambda p: int(p.stem.split("sec")[1]))
        if not sec_files:
            continue

        inf.reset()
        header = None
        confirm_sec = None
        last_prob = None
        last_pred = 0
        last_raw_vals = {}

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

            values = []
            for v in data_row:
                try:
                    val = float(v)
                    if not np.isfinite(val):
                        val = 0.0
                except:
                    val = 0.0
                values.append(val)
            x = np.array(values, dtype=np.float32)

            pred, prob, is_confirmed = inf.process_sec(x, sec, header)
            last_pred = pred
            last_prob = prob
            last_raw_vals = {col_names[i]: values[i] for i in range(min(len(col_names), len(values)))}

            if is_confirmed and confirm_sec is None:
                confirm_sec = sec

        confirmed_label = inf.confirmed_label or LABELS[last_pred]
        ans = answers.get(tid, "?")

        if confirmed_label == ans:
            correct += 1
        else:
            conf = last_prob[last_pred] if last_prob is not None else 0
            # top2 예측
            top2_idx = np.argsort(last_prob)[::-1][:2] if last_prob is not None else [0, 0]
            top2 = [(LABELS[i], last_prob[i]) for i in top2_idx]

            errors.append({
                "tid": tid, "ans": ans, "pred": confirmed_label,
                "conf": conf, "sec": confirm_sec,
                "top2": top2,
                "pprz": last_raw_vals.get("PPRZ", 0),
                "pctmt": last_raw_vals.get("PCTMT", 0),
                "vsump": last_raw_vals.get("VSUMP", 0),
            })

    acc = correct / total * 100
    print(f"\n{'='*80}")
    print(f"  V3 5ep: {correct}/{total} ({acc:.1f}%)")
    print(f"{'='*80}")

    # 클래스별 정확도
    class_c, class_t = defaultdict(int), defaultdict(int)
    # 다시 돌리지 않고 오답에서 계산
    for tid in range(1, 201):
        ans = answers.get(tid, "?")
        folder = DATA_ROOT / f"test{tid}"
        if not folder.exists():
            continue
        class_t[ans] += 1
        if tid not in [e["tid"] for e in errors]:
            class_c[ans] += 1

    print(f"\n  [클래스별 정확도]")
    for label in LABELS:
        t = class_t.get(label, 0)
        c = class_c.get(label, 0)
        if t > 0:
            print(f"    {label:>12s}: {c}/{t} ({c/t*100:.1f}%)")

    # 오답 상세
    if errors:
        print(f"\n  [오답 상세] ({len(errors)}건)")
        for e in errors:
            sec_str = f"sec{e['sec']}" if e['sec'] else "---"
            t1, t2 = e["top2"]
            print(f"    test{e['tid']:>3d}: {e['ans']:>12s} -> {e['pred']:>12s}  "
                  f"conf={e['conf']:.4f}  {sec_str}")
            print(f"             top2: {t1[0]}({t1[1]:.4f}), {t2[0]}({t2[1]:.4f})")
            print(f"             PPRZ={e['pprz']/1e6:.2f}M  PCTMT={e['pctmt']:.0f}  VSUMP={e['vsump']:.0f}")


if __name__ == "__main__":
    main()
