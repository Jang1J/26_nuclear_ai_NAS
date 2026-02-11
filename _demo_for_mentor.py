"""
멘토님 검증용: Team N Code main.py 로직 그대로 사용하여 1~60초 출력 데모.
실제 UDP 전송 대신 콘솔 출력만 수행.
"""
import os, sys, csv, time
import warnings
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Team N code의 main.py 모듈 import
TEAM_N_PY = Path('/Users/jangjaewon/Desktop/NAS/Team N code/py')
sys.path.insert(0, str(TEAM_N_PY))
from main import LABELS, RealtimeInference, load_pipeline, sanitize_array

# ===== 모델 로딩 =====
print("=" * 70)
print("  Team 6 — 멘토님 검증용 1~60초 출력 데모")
print("=" * 70)

model, scaler, feat_transformer, raw_feature_names = load_pipeline()
inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)

# 워밍업
dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)), dtype=np.float32)
_ = model.predict(dummy, verbose=0)
print("[INIT] Warmup done.\n")

# ===== 테스트 케이스 선택 =====
TEST_ID = 61
DATA_DIR = Path('/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100')

# 정답 확인
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['test_id']) == TEST_ID:
            answer_label = row['label']
            answer_delay = int(row['malf_delay'])
            break

print(f"[INFO] Test Case: test{TEST_ID}")
print(f"[INFO] 정답: {answer_label}, 사고지연: {answer_delay}초")
print(f"[INFO] 설정: CONFIRM={inferencer.CONFIRM_COUNT}, LOCA_CONFIRM={inferencer.LOCA_CONFIRM_COUNT}, GRACE={inferencer.GRACE_PERIOD}")
print(f"[INFO] LATE_WINDOW={inferencer.LATE_WINDOW}, LATE_CONF={inferencer.LATE_CONF}")
print()

# ===== 실제 main.py와 동일한 로직으로 1~60초 처리 =====
inferencer.reset()
diagnostic_results = None
diagnostic_time = None
header = None

print(f"{'sec':>4s} | {'diagnostic_results':>18s} | {'diagnostic_time':>16s} | {'prob_1,...,prob_9 (9 classes)':s}")
print("-" * 120)

for sec in range(1, 61):
    file_path = DATA_DIR / f"test{TEST_ID}_sec{sec}.csv"

    if not file_path.exists():
        print(f"{sec:>4d} | {'(file not found)':>18s} |")
        continue

    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        continue

    col_names = rows[0]
    data_row = rows[1]

    # ===== 파이프라인 시간 측정 (멘토님 코드와 동일 위치) =====
    start_time = time.time()

    # 안전한 float 변환
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

    # 전처리 + 모델 추론 + 진단 확정
    if header is None:
        header = col_names

    pred, prob, is_confirmed = inferencer.process_sec(x, sec, header)

    runtime = time.time() - start_time

    # 진단 확정 시 1회 수행
    if diagnostic_results is None and is_confirmed:
        diagnostic_results = LABELS[pred]
        diagnostic_time = round(sec + runtime, 1)

    # Late Correction
    if diagnostic_results is not None:
        corrected = inferencer.get_late_corrected_label()
        if corrected != diagnostic_results:
            diagnostic_results = corrected

    # ===== UDP 전송 포맷과 동일하게 출력 =====
    probs_str = ",".join(f"{float(p):.6f}" for p in prob)
    dr = diagnostic_results if diagnostic_results is not None else "None"
    dt = diagnostic_time if diagnostic_time is not None else 0.0

    # 실제 UDP 메시지 형식: test{n} sec{t}|diagnostic_results,diagnostic_time,prob_1,...,prob_9
    udp_msg = f"test{TEST_ID} sec{sec}|{dr},{dt},{probs_str}"

    # 콘솔에 보기좋게 출력
    prob_short = ",".join(f"{float(p):.6f}" for p in prob)
    marker = ""
    if is_confirmed:
        marker = f" <== 진단확정: {diagnostic_results}"
    dt_str = str(dt)
    print(f"{sec:>4d} | {dr:>18s} | {dt_str:>16s} | {prob_short}{marker}")

print()
print("=" * 70)
print(f"  최종 진단: {diagnostic_results if diagnostic_results else 'NORMAL'}")
print(f"  정답:      {answer_label}")
print(f"  결과:      {'O 정답' if (diagnostic_results or 'NORMAL') == answer_label else 'X 오답'}")
if diagnostic_time:
    react = diagnostic_time - answer_delay
    print(f"  확정시간:  {diagnostic_time}s (사고 후 {react:.1f}초 만에 감지)")
print("=" * 70)
