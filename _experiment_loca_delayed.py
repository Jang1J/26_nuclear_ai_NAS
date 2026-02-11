"""
조교님 요구사항 반영:
- "처음 보낸 게 최종 진단" → Late Correction 제거
- LOCA 확정 조건 강화: 연속 N초를 늘리거나, LOCA 내 순수성 확인
- SGTR/ESDE는 그대로 2연속

전략: LOCA 확정 시 "최근 M초 중 모두 같은 LOCA 하위 클래스"일 때만 확정
→ M = 6, 7, 8, 10 등 시뮬레이션
"""
import os, sys, csv, warnings
from pathlib import Path
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

TEAM_N_PY = Path('/Users/jangjaewon/Desktop/NAS/Team N code/py')
sys.path.insert(0, str(TEAM_N_PY))
from main import LABELS, RealtimeInference, load_pipeline

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

ALL_TIDS = sorted(answers.keys())

# 여러 LOCA_CONFIRM_COUNT 시뮬레이션 (Late Correction 없이)
GRACE_PERIOD = 3
CONFIRM_COUNT = 2  # SGTR/ESDE
LOCA_INDICES = {1, 2, 3}

for LOCA_N in [4, 5, 6, 7, 8, 10]:
    correct = 0
    wrong = 0
    unconfirmed = 0
    wrong_list = []

    for tid in ALL_TIDS:
        data_dir = find_data_dir(tid)
        if data_dir is None:
            continue

        inferencer.reset()
        header = None
        all_preds = []

        confirmed_label = None
        confirmed_sec = None

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

            pred, prob, _ = inferencer.process_sec(x, sec, header)
            all_preds.append(pred)

            if confirmed_label is not None:
                continue  # 이미 확정

            if pred == 0:
                continue
            if len(all_preds) <= GRACE_PERIOD:
                continue

            # 확정 조건
            if pred in LOCA_INDICES:
                n_req = LOCA_N
            else:
                n_req = CONFIRM_COUNT

            if len(all_preds) >= n_req:
                recent = all_preds[-n_req:]
                if all(p == pred for p in recent):
                    confirmed_label = LABELS[pred]
                    confirmed_sec = sec

        # Late Correction 없음!
        final = confirmed_label if confirmed_label else 'NORMAL'
        true_label = answers[tid]['label']

        if final == true_label:
            correct += 1
        elif confirmed_label is None and true_label == 'NORMAL':
            correct += 1
        elif confirmed_label is None:
            unconfirmed += 1
            wrong_list.append(f"test{tid}({true_label},d={answers[tid]['delay']},미확정)")
        else:
            wrong += 1
            wrong_list.append(f"test{tid}({true_label}→{final},d={answers[tid]['delay']})")

    total = correct + wrong + unconfirmed
    print(f"LOCA_CONFIRM={LOCA_N:>2d} (LC없음) | {correct}/{total} 정답 | 오답={wrong} | 미확정={unconfirmed}")
    if wrong_list:
        for w in wrong_list:
            print(f"    {w}")
    print()
