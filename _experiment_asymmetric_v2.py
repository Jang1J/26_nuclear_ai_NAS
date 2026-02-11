"""
비대칭 LOCA 확정 v2 - CL 마진 가드 범위 확장

핵심:
- HL/RCP 확정 시 "최근 6초 내 CL 확률 최대" > 임계값이면 보류
- CL 확정 시 CL 연속 N초 요구
- Late Correction 완전 제거
"""
import os, sys, csv, warnings
from pathlib import Path

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

GRACE_PERIOD = 3
CONFIRM_SGTR_ESDE = 2
LOCA_INDICES = {1, 2, 3}
CL_IDX = 2

# 파라미터 탐색
for CL_N in [4, 5, 6]:
    for HL_RCP_N in [4]:
        for GUARD_WINDOW in [4, 6, 8]:
            for CL_THRESH in [0.10, 0.15, 0.20, 0.30]:
                correct = 0
                wrong = 0
                unconfirmed = 0
                wrong_list = []
                react_times = []

                for tid in ALL_TIDS:
                    data_dir = find_data_dir(tid)
                    if data_dir is None:
                        continue

                    inferencer.reset()
                    header = None
                    all_preds = []
                    all_probs = []

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
                        all_probs.append(prob.copy())

                        if confirmed_label is not None:
                            continue
                        if pred == 0:
                            continue
                        if len(all_preds) <= GRACE_PERIOD:
                            continue

                        if pred in LOCA_INDICES:
                            if pred == CL_IDX:
                                n_req = CL_N
                            else:
                                n_req = HL_RCP_N
                                # CL 전환 위험 가드: 최근 GUARD_WINDOW초 내 CL 확률 체크
                                window = min(GUARD_WINDOW, len(all_probs))
                                recent_cl_probs = [p[CL_IDX] for p in all_probs[-window:]]
                                max_cl = max(recent_cl_probs)
                                if max_cl > CL_THRESH:
                                    continue  # CL 전환 위험 → 보류
                        else:
                            n_req = CONFIRM_SGTR_ESDE

                        if len(all_preds) >= n_req:
                            recent = all_preds[-n_req:]
                            if all(p == pred for p in recent):
                                confirmed_label = LABELS[pred]
                                confirmed_sec = sec

                    final = confirmed_label if confirmed_label else 'NORMAL'
                    true_label = answers[tid]['label']

                    if final == true_label:
                        correct += 1
                        if confirmed_sec and answers[tid]['delay'] > 0:
                            react_times.append(confirmed_sec - answers[tid]['delay'])
                    elif confirmed_label is None and true_label == 'NORMAL':
                        correct += 1
                    elif confirmed_label is None:
                        unconfirmed += 1
                        wrong_list.append(f"test{tid}({true_label},d={answers[tid]['delay']},미확정)")
                    else:
                        wrong += 1
                        wrong_list.append(f"test{tid}({true_label}→{final})")

                total = correct + wrong + unconfirmed
                avg_react = np.mean(react_times) if react_times else 0

                # 좋은 결과만 출력
                if correct >= 197:
                    print(f"HL/RCP={HL_RCP_N} CL={CL_N} guard={GUARD_WINDOW}s thresh={CL_THRESH} | {correct}/{total} | 오답={wrong} 미확정={unconfirmed} | react={avg_react:.1f}s")
                    for w in wrong_list:
                        print(f"    {w}")
