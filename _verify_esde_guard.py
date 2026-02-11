"""
Team 6 code 최종 로직 검증 (CL Guard + ESDE Guard).
467개 전체 캐시로 시뮬레이션.
"""
import pickle, csv
import numpy as np

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]

# Team 6 code main.py와 동일한 파라미터
GRACE_PERIOD = 3
CONFIRM_COUNT = 2       # SGTR/ESDE
LOCA_CONFIRM_COUNT = 4  # LOCA
LOCA_INDICES = {1, 2, 3}
CL_IDX = 2
ESDE_INDICES = {7, 8}
SGTR_INDICES = {4, 5, 6}

CL_GUARD_WINDOW = 5
CL_GUARD_THRESH = 0.15
ESDE_GUARD_WINDOW = 3
ESDE_GUARD_THRESH = 0.05


def simulate_one(preds, probs):
    """Team 6 main.py process_sec 로직 그대로 재현."""
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

        # pred_history 길이 = i+1 (0부터 i까지)
        hist_len = i + 1

        if pred in LOCA_INDICES:
            n_confirm = LOCA_CONFIRM_COUNT  # 4
            if hist_len >= n_confirm:
                recent = preds[i+1-n_confirm:i+1]
                if all(p == pred for p in recent):
                    if pred != CL_IDX:
                        # CL Guard
                        w = min(CL_GUARD_WINDOW, hist_len)
                        recent_cl = [probs[j][CL_IDX] for j in range(max(0, i+1-w), i+1)]
                        if max(recent_cl) > CL_GUARD_THRESH:
                            pass  # 보류
                        else:
                            confirmed_label = LABELS[pred]
                            confirmed_sec = sec
                    else:
                        # CL 4연속 즉시 확정
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec

        elif pred in ESDE_INDICES:
            n_confirm = CONFIRM_COUNT  # 2
            if hist_len >= n_confirm:
                recent = preds[i+1-n_confirm:i+1]
                if all(p == pred for p in recent):
                    # ESDE Guard
                    w = min(ESDE_GUARD_WINDOW, hist_len)
                    recent_loca_max = 0
                    for j in range(max(0, i+1-w), i+1):
                        loca_sum = sum(probs[j][k] for k in LOCA_INDICES)
                        recent_loca_max = max(recent_loca_max, loca_sum)
                    if recent_loca_max > ESDE_GUARD_THRESH:
                        pass  # 보류
                    else:
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec

        else:
            # SGTR: 2연속
            n_confirm = CONFIRM_COUNT  # 2
            if hist_len >= n_confirm:
                recent = preds[i+1-n_confirm:i+1]
                if all(p == pred for p in recent):
                    confirmed_label = LABELS[pred]
                    confirmed_sec = sec

    final = confirmed_label if confirmed_label else 'NORMAL'
    return final, confirmed_sec


# ===== 데이터 로드 =====
datasets = []

# 1~200
with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl', 'rb') as f:
    c = pickle.load(f)
a = {}
with open('/Users/jangjaewon/Desktop/NAS/data/Real_test_data/answers.csv') as f:
    for row in csv.DictReader(f):
        a[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
datasets.append(('test1-200', c, a))

# 201~300
with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_201_300.pkl', 'rb') as f:
    c = pickle.load(f)
a = {}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/test201_300/answers.csv') as f:
    for row in csv.DictReader(f):
        a[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
datasets.append(('test201-300', c, a))

# CL 150
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

# dt5
with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_dt5.pkl', 'rb') as f:
    c = pickle.load(f)
a = {}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/test_dt5_/answers.csv') as f:
    for row in csv.DictReader(f):
        a[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
datasets.append(('dt5', c, a))

# ===== 시뮬레이션 =====
print("=" * 80)
print("  Team 6 최종 로직 검증: CL Guard + ESDE Guard")
print("  CL Guard:   GW=5, T=0.15")
print("  ESDE Guard: GW=3, T=0.05")
print("=" * 80)

total_correct = 0
total_all = 0
total_wrong_list = []

for ds_name, cache, answers in datasets:
    correct = 0
    wrong_list = []
    esde_reacts = []

    for tid in sorted(answers.keys()):
        if tid not in cache:
            continue
        preds = cache[tid]['preds']
        probs = cache[tid]['probs']
        true_label = answers[tid]['label']
        delay = answers[tid].get('delay', 0)

        final, conf_sec = simulate_one(preds, probs)
        is_correct = (final == true_label) or (final == 'NORMAL' and true_label == 'NORMAL')

        if is_correct:
            correct += 1
        else:
            info = f"test{tid}({true_label}"
            if 'leak_size' in answers[tid]:
                info += f",leak={answers[tid]['leak_size']},n={answers[tid]['node']}"
            elif delay > 0:
                info += f",d={delay}"
            info += f"→{final})"
            wrong_list.append(info)

        # ESDE 반응시간
        if 'ESDE' in true_label and is_correct and conf_sec and delay > 0:
            esde_reacts.append(conf_sec - delay)

    n = len([t for t in answers if t in cache])
    total_correct += correct
    total_all += n

    esde_str = f", ESDE react avg={np.mean(esde_reacts):.2f}s" if esde_reacts else ""
    print(f"\n  [{ds_name}] {correct}/{n}{esde_str}")
    if wrong_list:
        for w in wrong_list:
            print(f"    오답: {w}")
        total_wrong_list.extend(wrong_list)

print(f"\n{'='*80}")
print(f"  합계: {total_correct}/{total_all} ({100*total_correct/total_all:.1f}%)")
print(f"  총 오답: {len(total_wrong_list)}건")
for w in total_wrong_list:
    print(f"    {w}")
print(f"{'='*80}")
