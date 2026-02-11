"""
LOCA 전체 (HL/CL/RCP) 67건의 초기 예측 안정성 분석.
LOCA 4연속 확정 시점에서, 최근 N초 내 다른 LOCA 하위 클래스 혼재 여부 확인.
→ 새 로직: "최근 6초 내 다른 LOCA 하위 클래스 있으면 확정 보류" 적용 시 부작용 확인
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

# 모든 LOCA 케이스
LOCA_CASES = [(tid, info) for tid, info in answers.items()
               if info['label'] in ('LOCA_HL', 'LOCA_CL', 'LOCA_RCP')]
LOCA_CASES.sort()

STABILITY_WINDOW = 6  # 확정 시점에서 최근 N초 확인

print("=" * 100)
print(f"  LOCA 전체 안정성 분석 (최근 {STABILITY_WINDOW}초 내 다른 LOCA 하위 클래스 혼재 여부)")
print("=" * 100)
print(f"{'test':>6s} | {'정답':>10s} | {'delay':>5s} | {'확정':>10s} | {'확정sec':>7s} | {'최근6초':>30s} | {'혼재':>4s} | {'보류시확정':>10s}")
print("-" * 100)

results_current = {'correct': 0, 'wrong': 0, 'unconfirmed': 0}
results_new = {'correct': 0, 'wrong': 0, 'unconfirmed': 0}

for tid, info in LOCA_CASES:
    data_dir = find_data_dir(tid)
    if data_dir is None:
        continue

    inferencer.reset()
    header = None
    all_preds = []

    # 새 로직용: 확정 보류 시뮬레이션
    new_confirmed_label = None
    new_confirmed_sec = None

    old_confirmed_label = None
    old_confirmed_sec = None

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
        all_preds.append(pred)

        # 기존 로직: 확정
        if old_confirmed_label is None and is_confirmed:
            old_confirmed_label = LABELS[pred]
            old_confirmed_sec = sec

        # 새 로직: LOCA 확정 시 안정성 체크
        if new_confirmed_label is None and is_confirmed:
            if pred in {1, 2, 3}:  # LOCA 계열
                # 최근 STABILITY_WINDOW초 예측 확인
                recent = all_preds[-STABILITY_WINDOW:]
                loca_preds = [p for p in recent if p in {1, 2, 3}]
                unique_loca = set(loca_preds)
                if len(unique_loca) <= 1:
                    # 안정적 → 확정
                    new_confirmed_label = LABELS[pred]
                    new_confirmed_sec = sec
                # else: 혼재 → 보류
            else:
                # LOCA가 아닌 경우 그대로 확정
                new_confirmed_label = LABELS[pred]
                new_confirmed_sec = sec

        # 새 로직: 보류 중이면 매 초 안정성 재확인
        if new_confirmed_label is None and not is_confirmed and pred in {1, 2, 3}:
            # GRACE_PERIOD 이후, 연속 4초 같은 LOCA + 최근 6초 안정
            if len(all_preds) > 3:
                n_confirm = 4  # LOCA는 항상 4
                if len(all_preds) >= n_confirm:
                    recent_n = all_preds[-n_confirm:]
                    if all(p == pred for p in recent_n):
                        # 4연속 달성, 안정성 체크
                        recent_stab = all_preds[-STABILITY_WINDOW:]
                        loca_preds = [p for p in recent_stab if p in {1, 2, 3}]
                        unique_loca = set(loca_preds)
                        if len(unique_loca) <= 1:
                            new_confirmed_label = LABELS[pred]
                            new_confirmed_sec = sec

    # Late Correction 적용 (기존 로직)
    old_final = old_confirmed_label
    if old_confirmed_label and old_confirmed_label.startswith('LOCA'):
        # Late Correction 시뮬레이션
        loca_preds_late = [p for p in all_preds[-10:] if p in {1, 2, 3}]
        if loca_preds_late:
            counter = Counter(loca_preds_late)
            most_idx, most_count = counter.most_common(1)[0]
            if most_idx != LABELS.index(old_confirmed_label):
                ratio = most_count / len(loca_preds_late)
                if ratio >= 0.5:
                    old_final = LABELS[most_idx]

    new_final = new_confirmed_label if new_confirmed_label else 'NORMAL'

    # 결과
    correct_label = info['label']

    # 기존 결과 (Late Correction 포함)
    if old_final is None:
        results_current['unconfirmed'] += 1
        old_res = 'X미확정'
    elif old_final == correct_label:
        results_current['correct'] += 1
        old_res = 'O'
    else:
        results_current['wrong'] += 1
        old_res = f'X({old_final})'

    # 새 로직 결과
    if new_final == correct_label:
        results_new['correct'] += 1
        new_res = f'O@{new_confirmed_sec}'
    elif new_confirmed_label is None:
        results_new['unconfirmed'] += 1
        new_res = 'X미확정'
    else:
        results_new['wrong'] += 1
        new_res = f'X({new_final})'

    # 확정 시점 최근 6초 예측
    if old_confirmed_sec:
        idx = old_confirmed_sec - 1
        recent = all_preds[max(0,idx-STABILITY_WINDOW+1):idx+1]
        recent_labels = [LABELS[p] if p in {1,2,3} else 'N' for p in recent]
        recent_str = ','.join([l.replace('LOCA_','') for l in recent_labels])
        loca_unique = set(p for p in recent if p in {1,2,3})
        mixed = 'Y' if len(loca_unique) > 1 else 'N'
    else:
        recent_str = '-'
        mixed = '-'

    print(f"test{tid:>3d} | {correct_label:>10s} | {info['delay']:>5d} | {old_confirmed_label or '-':>10s} | {old_confirmed_sec or '-':>7} | {recent_str:>30s} | {mixed:>4s} | {new_res:>10s}")

print()
print("=" * 60)
print(f"기존 로직 (4연속+LC): 정답={results_current['correct']}, 오답={results_current['wrong']}, 미확정={results_current['unconfirmed']}")
print(f"새 로직 (안정성체크): 정답={results_new['correct']}, 오답={results_new['wrong']}, 미확정={results_new['unconfirmed']}")
print("=" * 60)
