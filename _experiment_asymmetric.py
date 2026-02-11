"""
비대칭 LOCA 확정 로직 실험 (Late Correction 제거)

핵심 원칙 (조교님):
- "처음 UDP로 보낸 게 최종 진단" → Late Correction 사용 불가
- 따라서 확정 전에 충분히 확인해야 함

전략:
- LOCA_HL/RCP: 4연속 + p(CL) 마진 체크 (CL 전환 위험 낮을 때만 확정)
- LOCA_CL: 6연속 (충분히 CL로 수렴한 뒤 확정)
- SGTR/ESDE: 2연속 (기존 유지)
- GRACE=3 (기존 유지)
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

GRACE_PERIOD = 3
CONFIRM_SGTR_ESDE = 2
LOCA_INDICES = {1, 2, 3}  # HL=1, CL=2, RCP=3
CL_IDX = 2

# 여러 CL 연속 요구 수 시뮬레이션
for CL_N in [4, 5, 6, 7, 8]:
    for HL_RCP_N in [4]:
        for CL_MARGIN in [0.0]:  # CL margin 체크는 별도
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

                    # 확정 조건 결정
                    if pred in LOCA_INDICES:
                        if pred == CL_IDX:
                            n_req = CL_N  # CL은 더 보수적
                        else:
                            n_req = HL_RCP_N  # HL/RCP는 기존대로

                            # 추가: HL/RCP 확정 시 CL 전환 위험 체크
                            # 최근 예측에서 CL 확률이 높으면 보류
                            if len(all_probs) >= 2:
                                recent_cl_probs = [p[CL_IDX] for p in all_probs[-3:]]
                                max_cl_recent = max(recent_cl_probs)
                                if max_cl_recent > 0.15:  # CL 확률 15% 이상이면 보류
                                    continue
                    else:
                        n_req = CONFIRM_SGTR_ESDE

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
                    if confirmed_sec and answers[tid]['delay'] > 0:
                        react_times.append(confirmed_sec - answers[tid]['delay'])
                elif confirmed_label is None and true_label == 'NORMAL':
                    correct += 1
                elif confirmed_label is None:
                    unconfirmed += 1
                    wrong_list.append(f"test{tid}({true_label},d={answers[tid]['delay']},미확정)")
                else:
                    wrong += 1
                    wrong_list.append(f"test{tid}({true_label}→{final},d={answers[tid]['delay']})")

            total = correct + wrong + unconfirmed
            avg_react = np.mean(react_times) if react_times else 0
            print(f"HL/RCP={HL_RCP_N} CL={CL_N} CL_margin_guard | {correct}/{total} | 오답={wrong} 미확정={unconfirmed} | avg_react={avg_react:.1f}s")
            if wrong_list:
                for w in wrong_list:
                    print(f"    {w}")
            print()
