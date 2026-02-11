"""
Team 6 code main.py의 실제 모델로 9개 클래스 대표 테스트 실행.
UDP 전송 없이 로그만 출력 — 대회 실행 시 찍히는 형태 그대로.
"""
import sys
import os
import csv
import time
import numpy as np
from pathlib import Path

# main.py 경로 추가
TEAM6_PY = Path("/Users/jangjaewon/Desktop/NAS/Team 6 code/py")
sys.path.insert(0, str(TEAM6_PY))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from main import load_pipeline, RealtimeInference, LABELS

# 테스트 케이스 정보 (Team 6 code/data에 복사해둔 것)
TEST_INFO = {
    1: {"label": "NORMAL",     "delay": 0},
    2: {"label": "LOCA_HL",    "delay": 31},
    3: {"label": "LOCA_CL",    "delay": 23},
    4: {"label": "LOCA_RCP",   "delay": 27},
    5: {"label": "SGTR_Loop1", "delay": 19},
    6: {"label": "SGTR_Loop2", "delay": 28},
    7: {"label": "SGTR_Loop3", "delay": 21},
    8: {"label": "ESDE_in",    "delay": 23},
    9: {"label": "ESDE_out",   "delay": 24},
}

DATA_ROOT = Path("/Users/jangjaewon/Desktop/NAS/Team 6 code/data")

# 모델 로드
model, scaler, feat_transformer, raw_feature_names = load_pipeline()
inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)

# 워밍업
dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)), dtype=np.float32)
_ = model.predict(dummy, verbose=0)
print("[INIT] Warmup done.\n")

for test_id in sorted(TEST_INFO.keys()):
    info = TEST_INFO[test_id]
    true_label = info["label"]
    delay = info["delay"]

    folder = DATA_ROOT / f"test{test_id}"
    if not folder.exists():
        print(f"[SKIP] test{test_id} 폴더 없음")
        continue

    # CSV 파일 개수 확인
    csv_files = sorted(folder.glob(f"test{test_id}_sec*.csv"))
    max_sec = len(csv_files)

    print(f"\n{'='*70}")
    print(f"  TEST {test_id} — 정답: {true_label} (delay={delay}s)")
    print(f"{'='*70}")

    inferencer.reset()
    diagnostic_results = None
    diagnostic_time = None
    header = None

    for sec in range(1, max_sec + 1):
        file_path = folder / f"test{test_id}_sec{sec}.csv"
        if not file_path.exists():
            break

        with open(file_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue

        col_names = rows[0]
        data_row = rows[1]

        start_time = time.time()

        values = []
        for v in data_row:
            try:
                val = float(v)
                if not np.isfinite(val):
                    val = 0.0
            except (ValueError, TypeError):
                val = 0.0
            values.append(val)
        x = np.array(values, dtype=np.float32)

        if header is None:
            header = col_names

        pred, prob, is_confirmed = inferencer.process_sec(x, sec, header)

        if diagnostic_results is None and is_confirmed:
            diagnostic_results = LABELS[pred]
            diagnostic_time = round(time.time() - start_time + sec, 1)
            print(f"  *** 진단 확정: {diagnostic_results} at {diagnostic_time}s ***")

        # UDP_SEND 형식 출력
        probs_str = ",".join(f"{float(p):.6f}" for p in prob)
        dr = diagnostic_results if diagnostic_results is not None else "None"
        dt = diagnostic_time if diagnostic_time is not None else 0.0
        msg = f"test{test_id} sec{sec}|{dr},{dt},{probs_str}"
        print(f"[UDP_SEND] {msg}")

    # 결과 요약
    final = diagnostic_results if diagnostic_results else "NORMAL"
    correct = "O" if final == true_label else "X"
    react = f"{diagnostic_time - delay:.1f}s" if diagnostic_time and delay > 0 else "-"
    print(f"  결과: {final} [{correct}] 반응시간: {react}")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
