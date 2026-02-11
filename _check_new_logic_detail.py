"""
최적 새 로직 (HL/RCP=4, CL=7, guard=8, thresh=0.05) 상세 분석
- delay < 50 정확도
- LOCA_CL 정답률
- HL/RCP/CL 각 반응시간 비교 (기존 LC 로직 vs 새 로직)
"""
import csv, pickle
from pathlib import Path
from collections import Counter
import numpy as np

CACHE_FILE = '/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl'

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

LOCA_INDICES = {1, 2, 3}
CL_IDX = 2
ALL_TIDS = sorted(answers.keys())


def simulate_lc():
    """기존 LC 로직: LOCA_CONFIRM=4, CONFIRM=2, GRACE=3, LC(LATE_WINDOW=10, LATE_CONF=0.6)"""
    GRACE = 3
    CONFIRM = 2
    LOCA_CONFIRM = 4
    LATE_WINDOW = 10
    LATE_CONF = 0.6

    results = []
    for tid in ALL_TIDS:
        if tid not in cache:
            continue
        preds = cache[tid]['preds']
        probs = cache[tid]['probs']

        confirmed_label = None
        confirmed_sec = None
        confirmed_idx = None
        late_corrected = False

        for i in range(len(preds)):
            sec = i + 1
            pred = preds[i]

            if confirmed_label is not None:
                if not late_corrected and confirmed_idx in LOCA_INDICES:
                    if i + 1 >= LATE_WINDOW:
                        recent_preds = preds[i+1-LATE_WINDOW:i+1]
                        recent_probs = probs[i+1-LATE_WINDOW:i+1]
                        loca_preds = [p for p in recent_preds if p in LOCA_INDICES]
                        if loca_preds:
                            counter = Counter(loca_preds)
                            most_idx, most_count = counter.most_common(1)[0]
                            if most_idx != confirmed_idx:
                                ratio = most_count / len(loca_preds)
                                avg_conf = np.mean([p[most_idx] for p in recent_probs])
                                if ratio >= 0.5 and avg_conf >= LATE_CONF:
                                    confirmed_label = LABELS[most_idx]
                                    confirmed_idx = most_idx
                                    late_corrected = True
                continue

            if pred == 0 or sec <= GRACE:
                continue

            n_req = LOCA_CONFIRM if pred in LOCA_INDICES else CONFIRM
            if i + 1 >= n_req:
                recent = preds[i+1-n_req:i+1]
                if all(p == pred for p in recent):
                    confirmed_label = LABELS[pred]
                    confirmed_idx = pred
                    confirmed_sec = sec

        final = confirmed_label if confirmed_label else 'NORMAL'
        true_label = answers[tid]['label']
        is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')
        results.append({'tid': tid, 'true_label': true_label, 'delay': answers[tid]['delay'],
                        'final': final, 'confirmed_sec': confirmed_sec, 'is_correct': is_correct})
    return results


def simulate_new():
    """새 로직: HL/RCP=4, CL=7, guard=8, thresh=0.05, LC 없음"""
    GRACE = 3
    CONFIRM = 2
    HL_RCP_N = 4
    CL_N = 7
    GUARD_WINDOW = 8
    CL_THRESH = 0.05

    results = []
    for tid in ALL_TIDS:
        if tid not in cache:
            continue
        preds = cache[tid]['preds']
        probs = cache[tid]['probs']

        confirmed_label = None
        confirmed_sec = None

        for i in range(len(preds)):
            sec = i + 1
            pred = preds[i]

            if confirmed_label is not None:
                continue
            if pred == 0 or sec <= GRACE:
                continue

            if pred in LOCA_INDICES:
                if pred == CL_IDX:
                    n_req = CL_N
                else:
                    n_req = HL_RCP_N
                    # CL 전환 가드
                    window = min(GUARD_WINDOW, i + 1)
                    recent_cl = [probs[j][CL_IDX] for j in range(max(0, i+1-window), i+1)]
                    max_cl = max(recent_cl)
                    if max_cl > CL_THRESH:
                        continue  # CL 위험 → 보류
            else:
                n_req = CONFIRM

            if i + 1 >= n_req:
                recent = preds[i+1-n_req:i+1]
                if all(p == pred for p in recent):
                    confirmed_label = LABELS[pred]
                    confirmed_sec = sec

        final = confirmed_label if confirmed_label else 'NORMAL'
        true_label = answers[tid]['label']
        is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')
        results.append({'tid': tid, 'true_label': true_label, 'delay': answers[tid]['delay'],
                        'final': final, 'confirmed_sec': confirmed_sec, 'is_correct': is_correct})
    return results


# 실행
results_lc = simulate_lc()
results_new = simulate_new()

for label, results in [("기존 LC 로직 (LOCA4연속 + LC교정)", results_lc),
                        ("새 로직 (HL/RCP=4, CL=7, guard=8, thresh=0.05, LC없음)", results_new)]:
    print("=" * 80)
    print(f"  {label}")
    print("=" * 80)

    # 전체
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    wrong_cases = [r for r in results if not r['is_correct']]
    print(f"  전체: {correct}/{total} ({100*correct/total:.1f}%)")
    for r in wrong_cases:
        print(f"    test{r['tid']}({r['true_label']} → {r['final']}, d={r['delay']})")

    # delay < 50
    lt50 = [r for r in results if r['delay'] < 50 or r['delay'] == 0]
    correct_lt50 = sum(1 for r in lt50 if r['is_correct'])
    wrong_lt50 = [r for r in lt50 if not r['is_correct']]
    print(f"\n  delay<50: {correct_lt50}/{len(lt50)} ({100*correct_lt50/len(lt50):.1f}%)")
    for r in wrong_lt50:
        print(f"    test{r['tid']}({r['true_label']} → {r['final']}, d={r['delay']})")

    # LOCA_CL
    cl_all = [r for r in results if r['true_label'] == 'LOCA_CL']
    cl_correct = sum(1 for r in cl_all if r['is_correct'])
    cl_wrong = [r for r in cl_all if not r['is_correct']]
    print(f"\n  LOCA_CL 전체: {cl_correct}/{len(cl_all)}")
    for r in cl_wrong:
        print(f"    test{r['tid']}(→{r['final']}, d={r['delay']})")

    cl_lt50 = [r for r in cl_all if r['delay'] < 50]
    cl_lt50_ok = sum(1 for r in cl_lt50 if r['is_correct'])
    cl_lt50_wrong = [r for r in cl_lt50 if not r['is_correct']]
    print(f"  LOCA_CL (d<50): {cl_lt50_ok}/{len(cl_lt50)}")
    for r in cl_lt50_wrong:
        print(f"    test{r['tid']}(→{r['final']}, d={r['delay']})")

    # 반응시간 by class
    print(f"\n  반응시간 (확정sec - delay):")
    for cls in ['LOCA_HL', 'LOCA_CL', 'LOCA_RCP', 'SGTR_Loop1', 'SGTR_Loop2', 'SGTR_Loop3', 'ESDE_in', 'ESDE_out']:
        rts = [r['confirmed_sec'] - r['delay'] for r in results
               if r['true_label'] == cls and r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]
        if rts:
            print(f"    {cls:>12s}: avg={np.mean(rts):.1f}s, min={min(rts)}, max={max(rts)} (N={len(rts)})")

    print()
