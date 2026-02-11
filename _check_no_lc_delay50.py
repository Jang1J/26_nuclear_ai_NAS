"""
LC 없이 기존 확정 로직(LOCA_CONFIRM=4, CONFIRM=2, GRACE=3)으로
delay < 50 케이스만 정확도 확인
"""
import csv, pickle
from pathlib import Path
from collections import Counter
import numpy as np

CACHE_FILE = '/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl'

# 정답 로드
answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}

with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

LABELS = ['NORMAL', 'LOCA_HL', 'LOCA_CL', 'LOCA_RCP',
          'SGTR_Loop1', 'SGTR_Loop2', 'SGTR_Loop3',
          'ESDE_in', 'ESDE_out']

GRACE_PERIOD = 3
CONFIRM_COUNT = 2
LOCA_CONFIRM = 4
LOCA_INDICES = {1, 2, 3}

ALL_TIDS = sorted(answers.keys())

# ===== LC 없는 기본 로직 =====
correct_all = 0
wrong_all = 0
unconfirmed_all = 0

correct_lt50 = 0
wrong_lt50 = 0
unconfirmed_lt50 = 0
wrong_list_lt50 = []

correct_ge50 = 0
wrong_ge50 = 0
unconfirmed_ge50 = 0
wrong_list_ge50 = []

for tid in ALL_TIDS:
    if tid not in cache:
        continue
    preds = cache[tid]['preds']
    probs = cache[tid]['probs']
    delay = answers[tid]['delay']
    true_label = answers[tid]['label']

    confirmed_label = None
    confirmed_sec = None

    for i in range(len(preds)):
        sec = i + 1
        pred = preds[i]

        if confirmed_label is not None:
            continue
        if pred == 0:
            continue
        if sec <= GRACE_PERIOD:
            continue

        # 확정 조건
        if pred in LOCA_INDICES:
            n_req = LOCA_CONFIRM
        else:
            n_req = CONFIRM_COUNT

        if i + 1 >= n_req:
            recent = preds[i+1-n_req:i+1]
            if all(p == pred for p in recent):
                confirmed_label = LABELS[pred]
                confirmed_sec = sec

    # Late Correction 없음!
    final = confirmed_label if confirmed_label else 'NORMAL'

    # 판정
    is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')

    if is_correct:
        correct_all += 1
    elif confirmed_label is None:
        unconfirmed_all += 1
    else:
        wrong_all += 1

    # delay < 50 vs >= 50
    if delay < 50 or delay == 0:  # delay=0 은 NORMAL
        if is_correct:
            correct_lt50 += 1
        elif confirmed_label is None:
            unconfirmed_lt50 += 1
            wrong_list_lt50.append(f"test{tid}({true_label}, d={delay}, 미확정)")
        else:
            wrong_lt50 += 1
            wrong_list_lt50.append(f"test{tid}({true_label} → {final}, d={delay})")
    else:
        if is_correct:
            correct_ge50 += 1
        elif confirmed_label is None:
            unconfirmed_ge50 += 1
            wrong_list_ge50.append(f"test{tid}({true_label}, d={delay}, 미확정)")
        else:
            wrong_ge50 += 1
            wrong_list_ge50.append(f"test{tid}({true_label} → {final}, d={delay})")

print("=" * 70)
print("  LC 없는 기본 로직 (LOCA_CONFIRM=4, CONFIRM=2, GRACE=3)")
print("=" * 70)

total_all = correct_all + wrong_all + unconfirmed_all
print(f"\n전체:       {correct_all}/{total_all} = {100*correct_all/total_all:.1f}%")
print(f"  오답={wrong_all}, 미확정={unconfirmed_all}")

total_lt50 = correct_lt50 + wrong_lt50 + unconfirmed_lt50
print(f"\ndelay < 50: {correct_lt50}/{total_lt50} = {100*correct_lt50/total_lt50:.1f}%")
print(f"  오답={wrong_lt50}, 미확정={unconfirmed_lt50}")
if wrong_list_lt50:
    for w in wrong_list_lt50:
        print(f"    {w}")

total_ge50 = correct_ge50 + wrong_ge50 + unconfirmed_ge50
if total_ge50 > 0:
    print(f"\ndelay >= 50: {correct_ge50}/{total_ge50} = {100*correct_ge50/total_ge50:.1f}%")
    print(f"  오답={wrong_ge50}, 미확정={unconfirmed_ge50}")
    if wrong_list_ge50:
        for w in wrong_list_ge50:
            print(f"    {w}")

# ===== 비교: LC 있는 기존 로직 =====
print("\n" + "=" * 70)
print("  비교: LC 있는 로직 (LATE_WINDOW=10, LATE_CONF=0.6)")
print("=" * 70)

LATE_WINDOW = 10
LATE_CONF = 0.6

correct_lc_lt50 = 0
wrong_lc_lt50 = 0
unconfirmed_lc_lt50 = 0
wrong_list_lc_lt50 = []

correct_lc_all = 0
wrong_lc_all = 0
unconfirmed_lc_all = 0

for tid in ALL_TIDS:
    if tid not in cache:
        continue
    preds = cache[tid]['preds']
    probs = cache[tid]['probs']
    delay = answers[tid]['delay']
    true_label = answers[tid]['label']

    confirmed_label = None
    confirmed_sec = None

    for i in range(len(preds)):
        sec = i + 1
        pred = preds[i]

        if confirmed_label is not None:
            # Late Correction 매 초 확인
            if confirmed_label.startswith('LOCA'):
                confirmed_idx = LABELS.index(confirmed_label)
                window_preds = preds[max(0, i+1-LATE_WINDOW):i+1]
                loca_preds = [p for p in window_preds if p in LOCA_INDICES]
                if loca_preds:
                    counter = Counter(loca_preds)
                    most_idx, most_count = counter.most_common(1)[0]
                    if most_idx != confirmed_idx:
                        ratio = most_count / len(loca_preds)
                        if ratio >= LATE_CONF:
                            confirmed_label = LABELS[most_idx]
            continue

        if pred == 0:
            continue
        if sec <= GRACE_PERIOD:
            continue

        if pred in LOCA_INDICES:
            n_req = LOCA_CONFIRM
        else:
            n_req = CONFIRM_COUNT

        if i + 1 >= n_req:
            recent = preds[i+1-n_req:i+1]
            if all(p == pred for p in recent):
                confirmed_label = LABELS[pred]
                confirmed_sec = sec

    final = confirmed_label if confirmed_label else 'NORMAL'
    is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')

    if is_correct:
        correct_lc_all += 1
    elif confirmed_label is None:
        unconfirmed_lc_all += 1
    else:
        wrong_lc_all += 1

    if delay < 50 or delay == 0:
        if is_correct:
            correct_lc_lt50 += 1
        elif confirmed_label is None:
            unconfirmed_lc_lt50 += 1
            wrong_list_lc_lt50.append(f"test{tid}({true_label}, d={delay}, 미확정)")
        else:
            wrong_lc_lt50 += 1
            wrong_list_lc_lt50.append(f"test{tid}({true_label} → {final}, d={delay})")

total_lc_all = correct_lc_all + wrong_lc_all + unconfirmed_lc_all
total_lc_lt50 = correct_lc_lt50 + wrong_lc_lt50 + unconfirmed_lc_lt50

print(f"\n전체:       {correct_lc_all}/{total_lc_all} = {100*correct_lc_all/total_lc_all:.1f}%")
print(f"delay < 50: {correct_lc_lt50}/{total_lc_lt50} = {100*correct_lc_lt50/total_lc_lt50:.1f}%")
if wrong_list_lc_lt50:
    for w in wrong_list_lc_lt50:
        print(f"    {w}")

print("\n" + "=" * 70)
print("  최종 비교")
print("=" * 70)
print(f"  LC 없음 delay<50: {correct_lt50}/{total_lt50} ({100*correct_lt50/total_lt50:.1f}%)")
print(f"  LC 있음 delay<50: {correct_lc_lt50}/{total_lc_lt50} ({100*correct_lc_lt50/total_lc_lt50:.1f}%)")
print(f"  차이: {correct_lc_lt50 - correct_lt50}개")
print("=" * 70)
