"""LC있음 vs 가드(LC없음) 클래스별 반응시간 비교"""
import csv, pickle
from collections import Counter
import numpy as np

with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl', 'rb') as f:
    cache1 = pickle.load(f)
with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_201_300.pkl', 'rb') as f:
    cache2 = pickle.load(f)
cache = {**cache1, **cache2}

answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    for row in csv.DictReader(f):
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/test201_300/answers.csv') as f:
    for row in csv.DictReader(f):
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}

LABELS = ['NORMAL','LOCA_HL','LOCA_CL','LOCA_RCP','SGTR_Loop1','SGTR_Loop2','SGTR_Loop3','ESDE_in','ESDE_out']
LOCA_INDICES = {1,2,3}; CL_IDX = 2
ALL_TIDS = sorted(answers.keys())

def sim_lc(tids):
    results = []
    for tid in tids:
        if tid not in cache: continue
        preds = cache[tid]['preds']; probs = cache[tid]['probs']
        cl = None; cs = None; ci = None; lc = False
        for i in range(len(preds)):
            s = i+1; p = preds[i]
            if cl is not None:
                if not lc and ci in LOCA_INDICES and i+1 >= 10:
                    rp = preds[i+1-10:i+1]; rpb = probs[i+1-10:i+1]
                    lp = [x for x in rp if x in LOCA_INDICES]
                    if lp:
                        c = Counter(lp); mi, mc = c.most_common(1)[0]
                        if mi != ci:
                            ratio = mc / len(lp)
                            avg_c = np.mean([x[mi] for x in rpb])
                            if ratio >= 0.5 and avg_c >= 0.6:
                                cl = LABELS[mi]; ci = mi; lc = True
                continue
            if p == 0 or s <= 3: continue
            n = 4 if p in LOCA_INDICES else 2
            if i+1 >= n and all(x == p for x in preds[i+1-n:i+1]):
                cl = LABELS[p]; ci = p; cs = s
        final = cl or 'NORMAL'; tl = answers[tid]['label']
        ok = (final == tl) or (cl is None and tl == 'NORMAL')
        results.append({'tid': tid, 'true_label': tl, 'delay': answers[tid]['delay'],
                        'final': final, 'confirmed_sec': cs, 'is_correct': ok})
    return results

def sim_guard(tids):
    results = []
    for tid in tids:
        if tid not in cache: continue
        preds = cache[tid]['preds']; probs = cache[tid]['probs']
        cl = None; cs = None
        for i in range(len(preds)):
            s = i+1; p = preds[i]
            if cl is not None: continue
            if p == 0 or s <= 3: continue
            if p in LOCA_INDICES:
                if p == CL_IDX:
                    n = 4
                else:
                    n = 4
                    w = min(4, i+1)
                    rcl = [probs[j][CL_IDX] for j in range(max(0, i+1-w), i+1)]
                    if max(rcl) > 0.35:
                        continue
            else:
                n = 2
            if i+1 >= n and all(x == p for x in preds[i+1-n:i+1]):
                cl = LABELS[p]; cs = s
        final = cl or 'NORMAL'; tl = answers[tid]['label']
        ok = (final == tl) or (cl is None and tl == 'NORMAL')
        results.append({'tid': tid, 'true_label': tl, 'delay': answers[tid]['delay'],
                        'final': final, 'confirmed_sec': cs, 'is_correct': ok})
    return results

r_lc = sim_lc(ALL_TIDS)
r_guard = sim_guard(ALL_TIDS)

print(f"{'클래스':>12s} | {'건수':>4s} | {'① LC있음':>12s} | {'③ 가드(LC없음)':>14s} | {'차이':>8s}")
print("-" * 72)

for cls in LABELS[1:]:
    rt_lc = [r['confirmed_sec'] - r['delay'] for r in r_lc
             if r['true_label'] == cls and r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]
    rt_gd = [r['confirmed_sec'] - r['delay'] for r in r_guard
             if r['true_label'] == cls and r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]

    n = max(len(rt_lc), len(rt_gd))
    lc_str = f"{np.mean(rt_lc):.1f}s (max {max(rt_lc)}s)" if rt_lc else "N/A"
    gd_str = f"{np.mean(rt_gd):.1f}s (max {max(rt_gd)}s)" if rt_gd else "N/A"

    if rt_lc and rt_gd:
        diff = np.mean(rt_gd) - np.mean(rt_lc)
        sign = "+" if diff > 0 else ""
        diff_str = f"{sign}{diff:.1f}s"
    else:
        diff_str = "-"

    print(f"{cls:>12s} | {n:>4d} | {lc_str:>12s} | {gd_str:>14s} | {diff_str:>8s}")

print("-" * 72)

# 전체 평균
all_lc = [r['confirmed_sec'] - r['delay'] for r in r_lc
          if r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]
all_gd = [r['confirmed_sec'] - r['delay'] for r in r_guard
          if r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]
diff_all = np.mean(all_gd) - np.mean(all_lc)
print(f"{'전체 평균':>12s} | {len(all_lc):>4d} | {np.mean(all_lc):.1f}s (max {max(all_lc)}s) | {np.mean(all_gd):.1f}s (max {max(all_gd)}s) | +{diff_all:.1f}s")
