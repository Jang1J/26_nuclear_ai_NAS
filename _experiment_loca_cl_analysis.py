"""
LOCA_CL 초기 오진 패턴 분석:
- 25건의 LOCA_CL에서 몇 건이 초기에 RCP/HL로 먼저 예측되는지
- Late Correction 없이 현재 로직(4연속)으로 어떤 결과가 나오는지
"""
import os, sys, csv, time, warnings
from pathlib import Path
from collections import Counter

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

# 정답 로드
answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = int(row['test_id'])
        answers[tid] = {'label': row['label'], 'delay': int(row['malf_delay'])}

# LOCA_CL 케이스
CL_CASES = [tid for tid, info in answers.items() if info['label'] == 'LOCA_CL']
CL_CASES.sort()

# 데이터 경로 찾기
def find_data_dir(tid):
    for d in ['/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100',
              '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200']:
        if Path(d).joinpath(f'test{tid}_sec1.csv').exists():
            return Path(d)
    return None

print("=" * 90)
print("  LOCA_CL 초기 오진 패턴 분석 (25건)")
print("=" * 90)
print(f"{'test':>6s} | {'delay':>5s} | {'첫LOCA예측':>10s} | {'4연속확정':>10s} | {'LC후':>10s} | {'최종1st':>8s} | 패턴")
print("-" * 90)

pattern_counts = {'바로CL': 0, 'HL먼저→LC교정': 0, 'RCP먼저→LC교정': 0, '미확정': 0, '오답': 0}

for tid in CL_CASES:
    data_dir = find_data_dir(tid)
    if data_dir is None:
        print(f"test{tid:>3d} | SKIP (데이터 없음)")
        continue

    ans = answers[tid]
    inferencer.reset()
    header = None

    first_loca_pred = None  # 처음으로 LOCA 계열 예측한 것
    confirmed_label = None
    confirmed_sec = None
    lc_label = None

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
                if not np.isfinite(val):
                    val = 0.0
            except (ValueError, TypeError):
                val = 0.0
            values.append(val)
        x = np.array(values, dtype=np.float32)

        if header is None:
            header = col_names

        pred, prob, is_confirmed = inferencer.process_sec(x, sec, header)

        # 첫 LOCA 예측 기록
        if first_loca_pred is None and pred in {1, 2, 3}:
            first_loca_pred = LABELS[pred]

        # 확정 기록 (Late Correction 전)
        if confirmed_label is None and is_confirmed:
            confirmed_label = LABELS[pred]
            confirmed_sec = sec

        # Late Correction 확인
        if confirmed_label is not None:
            corrected = inferencer.get_late_corrected_label()
            if corrected != confirmed_label and lc_label is None:
                lc_label = corrected
                confirmed_label_after_lc = corrected

    # 최종 결과
    final = lc_label if lc_label else confirmed_label if confirmed_label else 'NORMAL'

    # 패턴 분류
    if confirmed_label == 'LOCA_CL':
        pattern = '바로CL'
        pattern_counts['바로CL'] += 1
    elif confirmed_label == 'LOCA_HL' and lc_label == 'LOCA_CL':
        pattern = 'HL→LC→CL'
        pattern_counts['HL먼저→LC교정'] += 1
    elif confirmed_label == 'LOCA_RCP' and lc_label == 'LOCA_CL':
        pattern = 'RCP→LC→CL'
        pattern_counts['RCP먼저→LC교정'] += 1
    elif confirmed_label is None:
        pattern = '미확정(60초)'
        pattern_counts['미확정'] += 1
    elif final != 'LOCA_CL':
        pattern = f'오답({final})'
        pattern_counts['오답'] += 1
    else:
        pattern = f'기타({confirmed_label}→{lc_label})'

    fl = first_loca_pred or '-'
    cl = confirmed_label or '-'
    lc = lc_label or '-'
    ok = 'O' if final == 'LOCA_CL' else 'X'

    print(f"test{tid:>3d} | {ans['delay']:>5d} | {fl:>10s} | {cl:>10s} | {lc:>10s} | {ok:>8s} | {pattern}")

print()
print("=" * 50)
print("패턴 요약:")
for k, v in pattern_counts.items():
    print(f"  {k}: {v}건")
print("=" * 50)
