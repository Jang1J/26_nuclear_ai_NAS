"""
1~200 + 201~300 통합 300개 3가지 로직 비교
"""
import csv, pickle
from pathlib import Path
from collections import Counter
import numpy as np

# 캐시 로드
with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl', 'rb') as f:
    cache1 = pickle.load(f)
with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_201_300.pkl', 'rb') as f:
    cache2 = pickle.load(f)

# 정답 로드
answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    for row in csv.DictReader(f):
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/test201_300/answers.csv') as f:
    for row in csv.DictReader(f):
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}

cache = {**cache1, **cache2}
ALL_TIDS = sorted(answers.keys())

LABELS = ['NORMAL', 'LOCA_HL', 'LOCA_CL', 'LOCA_RCP',
          'SGTR_Loop1', 'SGTR_Loop2', 'SGTR_Loop3',
          'ESDE_in', 'ESDE_out']
LOCA_INDICES = {1, 2, 3}
CL_IDX = 2

print(f"총 {len(ALL_TIDS)}개 테스트")

# 클래스 분포
from collections import Counter as Cnt
dist = Cnt(v['label'] for v in answers.values())
print(f"클래스: {dict(sorted(dist.items()))}")


def sim_lc(tids):
    GRACE=3; CONFIRM=2; LOCA_CONFIRM=4; LATE_WINDOW=10; LATE_CONF=0.6
    results = []
    for tid in tids:
        if tid not in cache: continue
        preds=cache[tid]['preds']; probs=cache[tid]['probs']
        cl=None; cs=None; ci=None; lc=False
        for i in range(len(preds)):
            sec=i+1; pred=preds[i]
            if cl is not None:
                if not lc and ci in LOCA_INDICES and i+1>=LATE_WINDOW:
                    rp=preds[i+1-LATE_WINDOW:i+1]; rpb=probs[i+1-LATE_WINDOW:i+1]
                    lp=[p for p in rp if p in LOCA_INDICES]
                    if lp:
                        c=Counter(lp); mi,mc=c.most_common(1)[0]
                        if mi!=ci:
                            ratio=mc/len(lp); avg_c=np.mean([p[mi] for p in rpb])
                            if ratio>=0.5 and avg_c>=LATE_CONF:
                                cl=LABELS[mi]; ci=mi; lc=True
                continue
            if pred==0 or sec<=GRACE: continue
            n=LOCA_CONFIRM if pred in LOCA_INDICES else CONFIRM
            if i+1>=n and all(p==pred for p in preds[i+1-n:i+1]):
                cl=LABELS[pred]; ci=pred; cs=sec
        final=cl or 'NORMAL'; tl=answers[tid]['label']
        ok=(final==tl)or(cl is None and tl=='NORMAL')
        results.append({'tid':tid,'true_label':tl,'delay':answers[tid]['delay'],'final':final,'confirmed_sec':cs,'is_correct':ok})
    return results

def sim_no_lc(tids):
    GRACE=3; CONFIRM=2; LOCA_CONFIRM=4
    results = []
    for tid in tids:
        if tid not in cache: continue
        preds=cache[tid]['preds']
        cl=None; cs=None
        for i in range(len(preds)):
            sec=i+1; pred=preds[i]
            if cl is not None: continue
            if pred==0 or sec<=GRACE: continue
            n=LOCA_CONFIRM if pred in LOCA_INDICES else CONFIRM
            if i+1>=n and all(p==pred for p in preds[i+1-n:i+1]):
                cl=LABELS[pred]; cs=sec
        final=cl or 'NORMAL'; tl=answers[tid]['label']
        ok=(final==tl)or(cl is None and tl=='NORMAL')
        results.append({'tid':tid,'true_label':tl,'delay':answers[tid]['delay'],'final':final,'confirmed_sec':cs,'is_correct':ok})
    return results

def sim_guard(tids, CL_N=4, GW=4, CT=0.35):
    GRACE=3; CONFIRM=2; HRN=4
    results = []
    for tid in tids:
        if tid not in cache: continue
        preds=cache[tid]['preds']; probs=cache[tid]['probs']
        cl=None; cs=None
        for i in range(len(preds)):
            sec=i+1; pred=preds[i]
            if cl is not None: continue
            if pred==0 or sec<=GRACE: continue
            if pred in LOCA_INDICES:
                if pred==CL_IDX: n=CL_N
                else:
                    n=HRN
                    w=min(GW,i+1); rcl=[probs[j][CL_IDX] for j in range(max(0,i+1-w),i+1)]
                    if max(rcl)>CT: continue
            else: n=CONFIRM
            if i+1>=n and all(p==pred for p in preds[i+1-n:i+1]):
                cl=LABELS[pred]; cs=sec
        final=cl or 'NORMAL'; tl=answers[tid]['label']
        ok=(final==tl)or(cl is None and tl=='NORMAL')
        results.append({'tid':tid,'true_label':tl,'delay':answers[tid]['delay'],'final':final,'confirmed_sec':cs,'is_correct':ok})
    return results


def print_full(label, results):
    total=len(results); correct=sum(1 for r in results if r['is_correct'])
    lt50=[r for r in results if r['delay']<50 or r['delay']==0]
    c_lt50=sum(1 for r in lt50 if r['is_correct'])
    wrong_all=[r for r in results if not r['is_correct']]

    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    print(f"  전체: {correct}/{total} ({100*correct/total:.1f}%) | d<50: {c_lt50}/{len(lt50)} ({100*c_lt50/len(lt50):.1f}%)")

    for cls in LABELS:
        cases=[r for r in results if r['true_label']==cls]
        if not cases: continue
        ok=sum(1 for r in cases if r['is_correct'])
        rts=[r['confirmed_sec']-r['delay'] for r in cases if r['is_correct'] and r['confirmed_sec'] and r['delay']>0]
        rt_str=f"avg={np.mean(rts):.1f}s max={max(rts)}s" if rts else "N/A"
        mark="✅" if ok==len(cases) else "❌"
        print(f"    {cls:>12s}: {ok:>2d}/{len(cases):>2d} {mark} | {rt_str}")

    if wrong_all:
        print(f"\n  오답 목록:")
        for r in sorted(wrong_all, key=lambda x: x['tid']):
            print(f"    test{r['tid']:>3d}: {r['true_label']} → {r['final']} (d={r['delay']})")

    # LOCA_CL 요약
    cl_all=[r for r in results if r['true_label']=='LOCA_CL']
    cl_lt50=[r for r in cl_all if r['delay']<50]
    cl_ok=sum(1 for r in cl_all if r['is_correct'])
    cl_lt50_ok=sum(1 for r in cl_lt50 if r['is_correct'])
    print(f"\n  LOCA_CL: 전체 {cl_ok}/{len(cl_all)}, d<50 {cl_lt50_ok}/{len(cl_lt50)}")


# 실행
r1 = sim_lc(ALL_TIDS)
r2 = sim_no_lc(ALL_TIDS)
r3 = sim_guard(ALL_TIDS)

print_full("① 기존 LC 로직", r1)
print_full("② LC 없는 기본 (4연속만)", r2)
print_full("③ 단순 가드 (CL=4, guard=4, thresh=0.35, LC없음)", r3)

# 최종 요약 테이블
print(f"\n{'='*90}")
print(f"  300개 통합 최종 비교")
print(f"{'='*90}")
for label, r in [("① LC 있음", r1), ("② LC 없음 기본", r2), ("③ 가드 LC없음", r3)]:
    total=len(r); correct=sum(1 for x in r if x['is_correct'])
    lt50=[x for x in r if x['delay']<50 or x['delay']==0]
    c_lt50=sum(1 for x in lt50 if x['is_correct'])
    cl_all=[x for x in r if x['true_label']=='LOCA_CL']
    cl_ok=sum(1 for x in cl_all if x['is_correct'])
    cl_lt50=[x for x in cl_all if x['delay']<50]
    cl_lt50_ok=sum(1 for x in cl_lt50 if x['is_correct'])

    hl_rt=[x['confirmed_sec']-x['delay'] for x in r if x['true_label']=='LOCA_HL' and x['is_correct'] and x['confirmed_sec'] and x['delay']>0]
    cl_rt=[x['confirmed_sec']-x['delay'] for x in r if x['true_label']=='LOCA_CL' and x['is_correct'] and x['confirmed_sec'] and x['delay']>0]
    rcp_rt=[x['confirmed_sec']-x['delay'] for x in r if x['true_label']=='LOCA_RCP' and x['is_correct'] and x['confirmed_sec'] and x['delay']>0]

    print(f"  {label:>15s}: {correct}/{total} (d<50: {c_lt50}/{len(lt50)}) | CL: {cl_ok}/{len(cl_all)}(d<50: {cl_lt50_ok}/{len(cl_lt50)}) | HL={np.mean(hl_rt):.1f}s CL={np.mean(cl_rt):.1f}s RCP={np.mean(rcp_rt):.1f}s")
print(f"{'='*90}")
