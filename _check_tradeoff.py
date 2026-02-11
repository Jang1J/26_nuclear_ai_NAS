"""
CL 가드 임계값별 반응시간 + 정확도 트레이드오프 분석
- thresh를 높이면 HL/RCP가 빨라지지만 CL 오진 위험 증가
- 최적 절충점 찾기
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


def simulate_new(HL_RCP_N, CL_N, GUARD_WINDOW, CL_THRESH):
    GRACE = 3
    CONFIRM = 2
    results = []

    for tid in ALL_TIDS:
        if tid not in cache:
            continue
        preds = cache[tid]['preds']
        probs = cache[tid]['probs']
        delay = answers[tid]['delay']

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
                    window = min(GUARD_WINDOW, i + 1)
                    recent_cl = [probs[j][CL_IDX] for j in range(max(0, i+1-window), i+1)]
                    max_cl = max(recent_cl)
                    if max_cl > CL_THRESH:
                        continue
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
        results.append({'tid': tid, 'true_label': true_label, 'delay': delay,
                        'final': final, 'confirmed_sec': confirmed_sec, 'is_correct': is_correct})
    return results


def simulate_lc():
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


def get_stats(results):
    lt50 = [r for r in results if r['delay'] < 50 or r['delay'] == 0]
    correct_lt50 = sum(1 for r in lt50 if r['is_correct'])
    wrong_lt50 = [r for r in lt50 if not r['is_correct']]

    stats = {}
    for cls in ['LOCA_HL', 'LOCA_CL', 'LOCA_RCP']:
        rts = [r['confirmed_sec'] - r['delay'] for r in results
               if r['true_label'] == cls and r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]
        stats[cls] = {'avg': np.mean(rts) if rts else 0, 'max': max(rts) if rts else 0, 'n': len(rts)}

    # LOCA_CL d<50 정답률
    cl_lt50 = [r for r in results if r['true_label'] == 'LOCA_CL' and r['delay'] < 50]
    cl_lt50_ok = sum(1 for r in cl_lt50 if r['is_correct'])

    # 전체 비LOCA 반응시간
    non_loca_rts = [r['confirmed_sec'] - r['delay'] for r in results
                    if r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0
                    and not r['true_label'].startswith('LOCA')]

    return {
        'correct_lt50': correct_lt50,
        'total_lt50': len(lt50),
        'wrong_lt50': wrong_lt50,
        'cl_lt50': f"{cl_lt50_ok}/{len(cl_lt50)}",
        'loca_hl': stats['LOCA_HL'],
        'loca_cl': stats['LOCA_CL'],
        'loca_rcp': stats['LOCA_RCP'],
        'non_loca_avg': np.mean(non_loca_rts) if non_loca_rts else 0,
    }


# 기존 LC 로직
print("=" * 110)
print("  기존 LC 로직 (기준)")
print("=" * 110)
rlc = simulate_lc()
slc = get_stats(rlc)
print(f"  d<50: {slc['correct_lt50']}/{slc['total_lt50']} | CL(d<50): {slc['cl_lt50']}")
print(f"  HL: avg={slc['loca_hl']['avg']:.1f}s max={slc['loca_hl']['max']}s | CL: avg={slc['loca_cl']['avg']:.1f}s max={slc['loca_cl']['max']}s | RCP: avg={slc['loca_rcp']['avg']:.1f}s max={slc['loca_rcp']['max']}s")
print(f"  비LOCA avg: {slc['non_loca_avg']:.1f}s")
for r in slc['wrong_lt50']:
    print(f"    틀림: test{r['tid']}({r['true_label']} → {r['final']}, d={r['delay']})")

# 다양한 파라미터 탐색
print()
print("=" * 110)
print(f"  {'파라미터':>35s} | {'d<50':>7s} | {'CL(d<50)':>8s} | {'HL avg':>6s} {'HL max':>6s} | {'CL avg':>6s} {'CL max':>6s} | {'RCP avg':>7s} {'RCP max':>7s} | 틀린케이스")
print("-" * 110)

for CL_N in [4, 5, 6, 7]:
    for GUARD_WINDOW in [4, 6, 8]:
        for CL_THRESH in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            results = simulate_new(4, CL_N, GUARD_WINDOW, CL_THRESH)
            s = get_stats(results)

            if s['correct_lt50'] < 185:
                continue

            param_str = f"CL={CL_N} guard={GUARD_WINDOW} thresh={CL_THRESH:.2f}"
            wrong_str = ", ".join([f"t{r['tid']}({r['true_label'][-2:]}→{r['final'][-2:]})" for r in s['wrong_lt50']])
            if not wrong_str:
                wrong_str = "없음 ✅"

            marker = ""
            if s['correct_lt50'] == s['total_lt50']:
                marker = " ✅"

            print(f"  {param_str:>35s} | {s['correct_lt50']:>3d}/{s['total_lt50']}{marker:>2s} | {s['cl_lt50']:>8s} | {s['loca_hl']['avg']:>5.1f}s {s['loca_hl']['max']:>5d}s | {s['loca_cl']['avg']:>5.1f}s {s['loca_cl']['max']:>5d}s | {s['loca_rcp']['avg']:>6.1f}s {s['loca_rcp']['max']:>6d}s | {wrong_str}")

print()
print("=" * 110)
print("  참고: 기존 LC 로직 HL=3.1s, CL=4.9s, RCP=3.3s (d<50 187/187)")
print("=" * 110)
