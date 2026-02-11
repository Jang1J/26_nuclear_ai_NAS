"""
V3 5ep 모델의 LOCA_CL/HL 오답 분석.
각 LOCA 테스트 케이스의 마지막 sec에서 핵심 센서값을 비교.
"""
import os, sys, csv, warnings
from pathlib import Path

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
    # 정답 로드
    answers = {}
    with open(ANSWERS_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answers[int(row["test_id"])] = row["label"]

    # LOCA_CL과 LOCA_HL 케이스만 추출
    loca_tests = {tid: label for tid, label in answers.items()
                  if label in ("LOCA_CL", "LOCA_HL")}

    # 모델 로드
    model, scaler, feat, raw_names = load_pipeline()
    inferencer = RealtimeInference(model, scaler, feat, raw_names)

    # 워밍업
    dummy = np.zeros((1, inferencer.WINDOW, len(feat.feature_names)), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    # 핵심 센서 인덱스 (대회 데이터 컬럼에서)
    key_sensors = ["PPRZ", "PCTMT", "VSUMP", "UHOLEG1", "UHOLEG2", "UHOLEG3",
                   "PSG1", "PSG2", "PSG3"]

    print("=" * 100)
    print("  LOCA_CL vs LOCA_HL 분석 — V3 5ep 모델")
    print("=" * 100)
    print(f"{'test':>6s} | {'정답':>10s} | {'예측':>10s} | {'OK':>2s} | {'conf':>6s} | {'sec':>4s} | PPRZ | PCTMT | VSUMP | probHL | probCL")
    print("-" * 100)

    for tid in sorted(loca_tests.keys()):
        ans = loca_tests[tid]
        folder = DATA_ROOT / f"test{tid}"
        if not folder.exists():
            continue

        sec_files = [p for p in folder.glob(f"test{tid}_sec*.csv") if " " not in p.name]
        sec_files = sorted(sec_files, key=lambda p: int(p.stem.split("sec")[1]))
        if not sec_files:
            continue

        inferencer.reset()
        header = None
        confirm_sec = None
        last_prob = None
        last_pred = 0

        # 마지막 sec의 원본 센서값 저장
        last_raw = None

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

            pred, prob, is_confirmed = inferencer.process_sec(x, sec, header)
            last_pred = pred
            last_prob = prob
            last_raw = {col_names[i]: values[i] for i in range(len(col_names))}

            if is_confirmed and confirm_sec is None:
                confirm_sec = sec

        confirmed_label = inferencer.confirmed_label or LABELS[last_pred]
        ok = "O" if confirmed_label == ans else "X"
        conf = last_prob[last_pred] if last_prob is not None else 0.0

        # 핵심 센서값
        pprz = last_raw.get("PPRZ", 0) if last_raw else 0
        pctmt = last_raw.get("PCTMT", 0) if last_raw else 0
        vsump = last_raw.get("VSUMP", 0) if last_raw else 0

        # LOCA_HL(1)과 LOCA_CL(2) 확률
        prob_hl = last_prob[1] if last_prob is not None else 0
        prob_cl = last_prob[2] if last_prob is not None else 0

        sec_str = f"{confirm_sec}" if confirm_sec else "---"

        print(f"  {tid:>4d} | {ans:>10s} | {confirmed_label:>10s} | {ok:>2s} | {conf:.4f} | {sec_str:>4s} | "
              f"{pprz/1e6:.2f}M | {pctmt:.0f} | {vsump:.0f} | {prob_hl:.4f} | {prob_cl:.4f}")

    # LOCA_CL 오답 분석
    print("\n" + "=" * 100)
    print("  오답 패턴 분석")
    print("=" * 100)


if __name__ == "__main__":
    main()
