"""
LOCA_CL 오확정 3건 (test57, 72, 133) 초별 상세 분석
- 어떤 초에 어떤 LOCA 하위 클래스로 예측했는지
- 확정 시점에서 최근 N초 예측이 어떤 분포인지
"""
import os, sys, csv, warnings
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

model, scaler, feat_transformer, raw_feature_names = load_pipeline()
inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)

dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)), dtype=np.float32)
_ = model.predict(dummy, verbose=0)

answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}

def find_data_dir(tid):
    for d in ['/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100',
              '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200']:
        if Path(d).joinpath(f'test{tid}_sec1.csv').exists():
            return Path(d)
    return None

TARGETS = [57, 72, 133]

for tid in TARGETS:
    data_dir = find_data_dir(tid)
    ans = answers[tid]

    print(f"\n{'='*80}")
    print(f"  test{tid} | 정답: {ans['label']} | delay={ans['delay']}")
    print(f"{'='*80}")
    print(f" sec | pred         | HL_prob  | CL_prob  | RCP_prob | 비정상후연속")
    print("-" * 80)

    inferencer.reset()
    header = None
    abnormal_streak = []  # 비정상 예측 연속 기록

    for sec in range(1, 61):
        fp = data_dir / f"test{tid}_sec{sec}.csv"
        if not fp.exists():
            continue
        with open(fp, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue

        col_names = rows[0]
        data_row = rows[1]
        values = []
        for v in data_row:
            try:
                val = float(v)
                if not np.isfinite(val): val = 0.0
            except: val = 0.0
            values.append(val)
        x = np.array(values, dtype=np.float32)
        if header is None:
            header = col_names

        pred, prob, is_confirmed = inferencer.process_sec(x, sec, header)

        # LOCA 확률만 추출
        hl_p = prob[1]
        cl_p = prob[2]
        rcp_p = prob[3]

        if pred != 0:
            abnormal_streak.append(LABELS[pred])
        else:
            abnormal_streak = []

        # 사고 후 데이터만 출력
        if sec >= ans['delay'] - 1:
            streak_str = ','.join(abnormal_streak[-6:]) if abnormal_streak else '-'
            marker = ' <<<< CONFIRMED' if is_confirmed else ''
            print(f" {sec:>3d} | {LABELS[pred]:>12s} | {hl_p:.6f} | {cl_p:.6f} | {rcp_p:.6f} | {streak_str}{marker}")

print()
