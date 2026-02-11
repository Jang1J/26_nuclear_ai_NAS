"""
Team 6 code main.py의 Fast Confirm 로직이 실험 결과와 일치하는지 검증.
main.py의 process_sec 로직을 그대로 재현하여 467개 테스트 시뮬레이션.
기대: 459/467 정답 (기존과 동일), ESDE/SGTR 반응시간 개선.
"""
import pickle, csv
import numpy as np

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]

# main.py의 상수 그대로 복사
WINDOW = 3
CONFIRM_COUNT = 2
LOCA_CONFIRM_COUNT = 4
GRACE_PERIOD = 3
LOCA_INDICES = {1, 2, 3}
CL_IDX = 2
ESDE_INDICES = {7, 8}
CL_GUARD_WINDOW = 5
CL_GUARD_THRESH = 0.15
ESDE_GUARD_WINDOW = 3
ESDE_GUARD_THRESH = 0.05
FAST_CONFIRM_THRESH = 0.95


def simulate_main_py(preds, probs):
    """main.py의 process_sec 확정 로직을 정확히 재현."""
    confirmed_label = None
    confirmed_sec = None
    pred_history = []
    prob_history = []

    for i in range(len(preds)):
        sec = i + 1
        pred = preds[i]
        prob = probs[i]

        pred_history.append(pred)
        prob_history.append(prob.copy())

        if confirmed_label is not None:
            continue
        if pred == 0:
            continue
        if len(pred_history) <= GRACE_PERIOD:
            continue

        if pred in LOCA_INDICES:
            n_confirm = LOCA_CONFIRM_COUNT
            if len(pred_history) >= n_confirm:
                recent = pred_history[-n_confirm:]
                if all(p == pred for p in recent):
                    if pred != CL_IDX:
                        w = min(CL_GUARD_WINDOW, len(prob_history))
                        recent_cl = [prob_history[-j-1][CL_IDX] for j in range(w)]
                        max_cl_prob = max(recent_cl)
                        if max_cl_prob <= CL_GUARD_THRESH:
                            confirmed_label = LABELS[pred]
                            confirmed_sec = sec
                    else:
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec

        elif pred in ESDE_INDICES:
            def _esde_guard_ok():
                w = min(ESDE_GUARD_WINDOW, len(prob_history))
                recent_loca_max = 0
                for j in range(1, w + 1):
                    loca_sum = sum(prob_history[-j][k] for k in LOCA_INDICES)
                    recent_loca_max = max(recent_loca_max, loca_sum)
                return recent_loca_max <= ESDE_GUARD_THRESH

            if prob[pred] >= FAST_CONFIRM_THRESH:
                if _esde_guard_ok():
                    confirmed_label = LABELS[pred]
                    confirmed_sec = sec
            else:
                n_confirm = CONFIRM_COUNT
                if len(pred_history) >= n_confirm:
                    recent = pred_history[-n_confirm:]
                    if all(p == pred for p in recent):
                        if _esde_guard_ok():
                            confirmed_label = LABELS[pred]
                            confirmed_sec = sec
        else:
            # SGTR
            if prob[pred] >= FAST_CONFIRM_THRESH:
                confirmed_label = LABELS[pred]
                confirmed_sec = sec
            else:
                n_confirm = CONFIRM_COUNT
                if len(pred_history) >= n_confirm:
                    recent = pred_history[-n_confirm:]
                    if all(p == pred for p in recent):
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec

    final = confirmed_label if confirmed_label else 'NORMAL'
    return final, confirmed_sec


# ===== 데이터 로드 =====
datasets = []

with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl', 'rb') as f:
    c = pickle.load(f)
a = {}
with open('/Users/jangjaewon/Desktop/NAS/data/Real_test_data/answers.csv') as f:
    for row in csv.DictReader(f):
        a[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
datasets.append(('test1-200', c, a))

with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_201_300.pkl', 'rb') as f:
    c = pickle.load(f)
a = {}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/test201_300/answers.csv') as f:
    for row in csv.DictReader(f):
        a[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
datasets.append(('test201-300', c, a))

with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_cl150.pkl', 'rb') as f:
    c = pickle.load(f)
a = {}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/LOCA_CL_test/answers.csv') as f:
    for row in csv.DictReader(f):
        a[int(row['test_id'])] = {
            'label': row['label'], 'delay': 0,
            'leak_size': int(row['leak_size']), 'node': int(row['node']),
        }
datasets.append(('CL150', c, a))

with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_dt5.pkl', 'rb') as f:
    c = pickle.load(f)
a = {}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/test_dt5_/answers.csv') as f:
    for row in csv.DictReader(f):
        a[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
datasets.append(('dt5', c, a))


# ===== 시뮬레이션 =====
print("=" * 80)
print("  Team 6 code main.py Fast Confirm 검증 (467개)")
print("  FAST_CONFIRM_THRESH = 0.95")
print("=" * 80)

total_correct = 0
total_all = 0
wrong_list = []
esde_reacts = []
sgtr_reacts = []
loca_reacts = {'HL': [], 'CL': [], 'RCP': []}

for ds_name, cache, answers in datasets:
    ds_correct = 0
    ds_total = 0
    for tid in sorted(answers.keys()):
        if tid not in cache:
            continue
        preds = cache[tid]['preds']
        probs_arr = cache[tid]['probs']
        true_label = answers[tid]['label']
        delay = answers[tid].get('delay', 0)

        final, conf_sec = simulate_main_py(preds, probs_arr)
        is_correct = (final == true_label)

        total_all += 1
        ds_total += 1
        if is_correct:
            total_correct += 1
            ds_correct += 1
        else:
            wrong_list.append(f"{ds_name}/test{tid}({true_label}→{final})")

        if is_correct and conf_sec and delay > 0:
            react = conf_sec - delay
            if 'ESDE' in true_label:
                esde_reacts.append(react)
            elif 'SGTR' in true_label:
                sgtr_reacts.append(react)
            elif 'LOCA_HL' in true_label:
                loca_reacts['HL'].append(react)
            elif 'LOCA_CL' in true_label:
                loca_reacts['CL'].append(react)
            elif 'LOCA_RCP' in true_label:
                loca_reacts['RCP'].append(react)

    print(f"  {ds_name}: {ds_correct}/{ds_total}")

print(f"\n  총합: {total_correct}/{total_all} ({100*total_correct/total_all:.1f}%)")

if esde_reacts:
    print(f"  ESDE avg react: {np.mean(esde_reacts):.2f}s ({len(esde_reacts)}건)")
if sgtr_reacts:
    print(f"  SGTR avg react: {np.mean(sgtr_reacts):.2f}s ({len(sgtr_reacts)}건)")
for k in ['HL', 'CL', 'RCP']:
    if loca_reacts[k]:
        print(f"  LOCA_{k} avg react: {np.mean(loca_reacts[k]):.2f}s ({len(loca_reacts[k])}건)")

if wrong_list:
    print(f"\n  오답 ({len(wrong_list)}건):")
    for w in wrong_list:
        print(f"    {w}")

# 기대값 체크
if total_correct == 459 and total_all == 467:
    print(f"\n  ✓ 검증 통과! 459/467 = 실험 결과와 일치")
else:
    print(f"\n  ✗ 불일치! 기대: 459/467, 실제: {total_correct}/{total_all}")

print("=" * 80)
