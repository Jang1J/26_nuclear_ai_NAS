"""
멘토님 검증용: 9개 클래스 각각 1건씩, 1~60초 전체 출력 데모.
Team N Code main.py 로직 그대로 사용.
"""
import os, sys, csv, time, warnings
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

TEAM_N_PY = Path('/Users/jangjaewon/Desktop/NAS/Team N code/py')
sys.path.insert(0, str(TEAM_N_PY))
from main import LABELS, RealtimeInference, load_pipeline, sanitize_array

# ===== 모델 로딩 =====
print("=" * 80)
print("  Team 6 — 9개 클래스 전체 데모 (멘토님 검증용)")
print("=" * 80)

model, scaler, feat_transformer, raw_feature_names = load_pipeline()
inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)

dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)), dtype=np.float32)
_ = model.predict(dummy, verbose=0)
print("[INIT] Warmup done.\n")

# ===== 정답 로드 =====
answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = int(row['test_id'])
        answers[tid] = {'label': row['label'], 'delay': int(row['malf_delay'])}

# ===== 9개 클래스별 테스트 케이스 =====
TEST_CASES = [
    (127, '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200'),  # NORMAL
    ( 53, '/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100'),    # LOCA_HL
    (133, '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200'),  # LOCA_CL
    ( 87, '/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100'),    # LOCA_RCP
    (178, '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200'),  # SGTR_Loop1
    (161, '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200'),  # SGTR_Loop2
    (173, '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200'),  # SGTR_Loop3
    (110, '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200'),  # ESDE_in
    ( 61, '/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100'),    # ESDE_out
]

for test_id, data_dir in TEST_CASES:
    ans = answers[test_id]
    data_path = Path(data_dir)

    print("\n" + "=" * 80)
    print(f"  test{test_id} | 정답: {ans['label']} | 사고지연: {ans['delay']}초")
    print("=" * 80)

    inferencer.reset()
    diagnostic_results = None
    diagnostic_time = None
    header = None

    print(f" {'sec':>3s} | {'diagnostic_results':>16s} | {'diag_time':>10s} | {'UDP: prob_1, ..., prob_9'}")
    print("-" * 110)

    for sec in range(1, 61):
        file_path = data_path / f"test{test_id}_sec{sec}.csv"
        if not file_path.exists():
            continue

        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue

        col_names = rows[0]
        data_row = rows[1]

        # ===== 파이프라인 시간 측정 (멘토님 코드와 동일 위치) =====
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
        runtime = time.time() - start_time

        if diagnostic_results is None and is_confirmed:
            diagnostic_results = LABELS[pred]
            diagnostic_time = round(sec + runtime, 1)

        if diagnostic_results is not None:
            corrected = inferencer.get_late_corrected_label()
            if corrected != diagnostic_results:
                diagnostic_results = corrected

        probs_str = ",".join(f"{float(p):.6f}" for p in prob)
        dr = diagnostic_results if diagnostic_results is not None else "None"
        dt = diagnostic_time if diagnostic_time is not None else 0.0
        dt_str = str(dt)

        marker = ""
        if is_confirmed:
            marker = f"  <== CONFIRMED"

        prob_short = ",".join(f"{float(p):.4f}" for p in prob)
        print(f" {sec:>3d} | {dr:>16s} | {dt_str:>10s} | {prob_short}{marker}")

    # 결과 요약
    final = diagnostic_results if diagnostic_results else "NORMAL"
    ok = "O" if final == ans['label'] else "X"
    react_str = ""
    if diagnostic_time and ans['delay'] > 0:
        react_str = f", 반응: {diagnostic_time - ans['delay']:.1f}s"
    print(f"  >>> 결과: [{ok}] 진단={final}, 정답={ans['label']}{react_str}")

print("\n" + "=" * 80)
print("  전체 완료")
print("=" * 80)
