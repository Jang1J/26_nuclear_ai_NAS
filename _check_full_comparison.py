"""
정확한 비교: LC 있음 vs LC 없음
- LC 로직을 main.py와 동일하게 재현 (ratio >= 0.5 AND avg_conf >= 0.6)
- delay < 50 기준 정확도
- LOCA_CL 25건 중 몇 개 맞추는지
- 각 로직별 평균 반응시간 비교
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

GRACE_PERIOD = 3
CONFIRM_COUNT = 2
LOCA_CONFIRM = 4
LOCA_INDICES = {1, 2, 3}
LATE_WINDOW = 10
LATE_CONF = 0.6

ALL_TIDS = sorted(answers.keys())


def simulate(use_lc):
    """하나의 로직으로 전체 시뮬레이션"""
    results = []

    for tid in ALL_TIDS:
        if tid not in cache:
            continue
        preds = cache[tid]['preds']
        probs = cache[tid]['probs']
        delay = answers[tid]['delay']
        true_label = answers[tid]['label']

        confirmed_label = None
        confirmed_sec = None
        confirmed_idx = None
        late_corrected = False

        for i in range(len(preds)):
            sec = i + 1
            pred = preds[i]
            prob = probs[i]

            if confirmed_label is not None:
                # LC 적용 (매 초)
                if use_lc and not late_corrected and confirmed_idx in LOCA_INDICES:
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
                    confirmed_idx = pred
                    confirmed_sec = sec

        final = confirmed_label if confirmed_label else 'NORMAL'
        is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')

        results.append({
            'tid': tid,
            'true_label': true_label,
            'delay': delay,
            'final': final,
            'confirmed_sec': confirmed_sec,
            'is_correct': is_correct,
        })

    return results


# 두 로직 실행
results_lc = simulate(use_lc=True)
results_no_lc = simulate(use_lc=False)

# ===== 출력 =====
for label, results in [("LC 있음", results_lc), ("LC 없음", results_no_lc)]:
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)

    # 전체
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    print(f"  전체: {correct}/{total} = {100*correct/total:.1f}%")

    # delay < 50
    lt50 = [r for r in results if r['delay'] < 50 or r['delay'] == 0]
    correct_lt50 = sum(1 for r in lt50 if r['is_correct'])
    wrong_lt50 = [r for r in lt50 if not r['is_correct']]
    print(f"  delay<50: {correct_lt50}/{len(lt50)} = {100*correct_lt50/len(lt50):.1f}%")
    for r in wrong_lt50:
        print(f"    test{r['tid']}({r['true_label']} → {r['final']}, d={r['delay']})")

    # LOCA_CL 전체
    cl_cases = [r for r in results if r['true_label'] == 'LOCA_CL']
    cl_correct = sum(1 for r in cl_cases if r['is_correct'])
    cl_wrong = [r for r in cl_cases if not r['is_correct']]
    print(f"  LOCA_CL: {cl_correct}/{len(cl_cases)}")
    for r in cl_wrong:
        print(f"    test{r['tid']}({r['true_label']} → {r['final']}, d={r['delay']})")

    # LOCA_CL delay < 50만
    cl_lt50 = [r for r in cl_cases if r['delay'] < 50]
    cl_lt50_correct = sum(1 for r in cl_lt50 if r['is_correct'])
    cl_lt50_wrong = [r for r in cl_lt50 if not r['is_correct']]
    print(f"  LOCA_CL (d<50): {cl_lt50_correct}/{len(cl_lt50)}")
    for r in cl_lt50_wrong:
        print(f"    test{r['tid']}({r['true_label']} → {r['final']}, d={r['delay']})")

    # 평균 반응시간 (사고 케이스만)
    react_all = []
    react_loca = []
    react_hl = []
    react_cl = []
    react_rcp = []
    for r in results:
        if r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0:
            rt = r['confirmed_sec'] - r['delay']
            react_all.append(rt)
            if r['true_label'].startswith('LOCA'):
                react_loca.append(rt)
            if r['true_label'] == 'LOCA_HL':
                react_hl.append(rt)
            elif r['true_label'] == 'LOCA_CL':
                react_cl.append(rt)
            elif r['true_label'] == 'LOCA_RCP':
                react_rcp.append(rt)

    print(f"  평균 반응시간:")
    print(f"    전체: {np.mean(react_all):.1f}s")
    print(f"    LOCA 전체: {np.mean(react_loca):.1f}s")
    print(f"    LOCA_HL: {np.mean(react_hl):.1f}s (N={len(react_hl)})")
    print(f"    LOCA_CL: {np.mean(react_cl):.1f}s (N={len(react_cl)})")
    print(f"    LOCA_RCP: {np.mean(react_rcp):.1f}s (N={len(react_rcp)})")
    print()
