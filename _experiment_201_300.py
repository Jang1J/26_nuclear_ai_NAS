"""
201~300 테스트 데이터로 3가지 로직 비교
1) 기존 LC 로직 (LOCA4연속 + LC교정)
2) 단순 가드 (CL=4, guard=4, thresh=0.35, LC없음)
3) Risk Gating (GPT 제안)
+ LC 없는 기본 (4연속만, 가드 없음)
"""
import os, sys, csv, pickle, time
from pathlib import Path
from collections import Counter
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

TEAM_N_PY = Path('/Users/jangjaewon/Desktop/NAS/Team N code/py')
sys.path.insert(0, str(TEAM_N_PY))
from main import LABELS, RealtimeInference, load_pipeline

DATA_DIR = Path('/Users/jangjaewon/Desktop/NAS/_test_data/test201_300')
ANSWERS_FILE = DATA_DIR / 'answers.csv'
CACHE_FILE = '/Users/jangjaewon/Desktop/NAS/_pred_cache_201_300.pkl'

# 정답 로드
answers = {}
with open(ANSWERS_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}

ALL_TIDS = sorted(answers.keys())
print(f"총 {len(ALL_TIDS)}개 테스트 (test{ALL_TIDS[0]}~test{ALL_TIDS[-1]})")

# 클래스 분포
from collections import Counter as Cnt
label_dist = Cnt(v['label'] for v in answers.values())
print(f"클래스 분포: {dict(sorted(label_dist.items()))}")
delay_dist = [v['delay'] for v in answers.values() if v['delay'] > 0]
print(f"delay 분포: min={min(delay_dist)}, max={max(delay_dist)}, avg={np.mean(delay_dist):.1f}")

# ===== 캐시 생성 =====
LOCA_INDICES = {1, 2, 3}
CL_IDX = 2

if not Path(CACHE_FILE).exists():
    print("\n[1/2] 모델 추론 캐시 생성 중...")
    model, scaler, feat_transformer, raw_feature_names = load_pipeline()
    inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)
    dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    cache = {}
    t0 = time.time()
    for i, tid in enumerate(ALL_TIDS):
        inferencer.reset()
        header = None
        preds = []
        probs = []

        for sec in range(1, 61):
            fp = DATA_DIR / f"test{tid}_sec{sec}.csv"
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
            preds.append(int(pred))
            probs.append(prob.tolist())

        cache[tid] = {'preds': preds, 'probs': probs}
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(ALL_TIDS)} done ({time.time()-t0:.0f}s)")

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"[1/2] 캐시 저장 완료 ({time.time()-t0:.0f}s)")
else:
    print(f"\n[1/2] 캐시 로드: {CACHE_FILE}")

with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)


# ===== 시뮬레이션 함수들 =====

def sim_lc(cache, answers, tids):
    """기존 LC 로직"""
    GRACE = 3; CONFIRM = 2; LOCA_CONFIRM = 4
    LATE_WINDOW = 10; LATE_CONF = 0.6
    results = []
    for tid in tids:
        if tid not in cache: continue
        preds = cache[tid]['preds']; probs = cache[tid]['probs']
        confirmed_label = None; confirmed_sec = None; confirmed_idx = None
        late_corrected = False
        for i in range(len(preds)):
            sec = i + 1; pred = preds[i]
            if confirmed_label is not None:
                if not late_corrected and confirmed_idx in LOCA_INDICES:
                    if i + 1 >= LATE_WINDOW:
                        rp = preds[i+1-LATE_WINDOW:i+1]
                        rpb = probs[i+1-LATE_WINDOW:i+1]
                        lp = [p for p in rp if p in LOCA_INDICES]
                        if lp:
                            c = Counter(lp); mi, mc = c.most_common(1)[0]
                            if mi != confirmed_idx:
                                ratio = mc / len(lp)
                                avg_c = np.mean([p[mi] for p in rpb])
                                if ratio >= 0.5 and avg_c >= LATE_CONF:
                                    confirmed_label = LABELS[mi]; confirmed_idx = mi
                                    late_corrected = True
                continue
            if pred == 0 or sec <= GRACE: continue
            n = LOCA_CONFIRM if pred in LOCA_INDICES else CONFIRM
            if i + 1 >= n:
                recent = preds[i+1-n:i+1]
                if all(p == pred for p in recent):
                    confirmed_label = LABELS[pred]; confirmed_idx = pred; confirmed_sec = sec
        final = confirmed_label or 'NORMAL'
        tl = answers[tid]['label']
        ok = (final == tl) or (confirmed_label is None and tl == 'NORMAL')
        results.append({'tid': tid, 'true_label': tl, 'delay': answers[tid]['delay'],
                        'final': final, 'confirmed_sec': confirmed_sec, 'is_correct': ok})
    return results


def sim_no_lc(cache, answers, tids):
    """LC 없는 기본 4연속"""
    GRACE = 3; CONFIRM = 2; LOCA_CONFIRM = 4
    results = []
    for tid in tids:
        if tid not in cache: continue
        preds = cache[tid]['preds']
        confirmed_label = None; confirmed_sec = None
        for i in range(len(preds)):
            sec = i + 1; pred = preds[i]
            if confirmed_label is not None: continue
            if pred == 0 or sec <= GRACE: continue
            n = LOCA_CONFIRM if pred in LOCA_INDICES else CONFIRM
            if i + 1 >= n:
                recent = preds[i+1-n:i+1]
                if all(p == pred for p in recent):
                    confirmed_label = LABELS[pred]; confirmed_sec = sec
        final = confirmed_label or 'NORMAL'
        tl = answers[tid]['label']
        ok = (final == tl) or (confirmed_label is None and tl == 'NORMAL')
        results.append({'tid': tid, 'true_label': tl, 'delay': answers[tid]['delay'],
                        'final': final, 'confirmed_sec': confirmed_sec, 'is_correct': ok})
    return results


def sim_guard(cache, answers, tids, CL_N=4, GUARD_W=4, CL_THRESH=0.35):
    """단순 가드 (LC없음)"""
    GRACE = 3; CONFIRM = 2; HL_RCP_N = 4
    results = []
    for tid in tids:
        if tid not in cache: continue
        preds = cache[tid]['preds']; probs = cache[tid]['probs']
        confirmed_label = None; confirmed_sec = None
        for i in range(len(preds)):
            sec = i + 1; pred = preds[i]
            if confirmed_label is not None: continue
            if pred == 0 or sec <= GRACE: continue
            if pred in LOCA_INDICES:
                if pred == CL_IDX:
                    n = CL_N
                else:
                    n = HL_RCP_N
                    window = min(GUARD_W, i + 1)
                    rcl = [probs[j][CL_IDX] for j in range(max(0, i+1-window), i+1)]
                    if max(rcl) > CL_THRESH:
                        continue
            else:
                n = CONFIRM
            if i + 1 >= n:
                recent = preds[i+1-n:i+1]
                if all(p == pred for p in recent):
                    confirmed_label = LABELS[pred]; confirmed_sec = sec
        final = confirmed_label or 'NORMAL'
        tl = answers[tid]['label']
        ok = (final == tl) or (confirmed_label is None and tl == 'NORMAL')
        results.append({'tid': tid, 'true_label': tl, 'delay': answers[tid]['delay'],
                        'final': final, 'confirmed_sec': confirmed_sec, 'is_correct': ok})
    return results


def print_results(label, results):
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    wrong = [r for r in results if not r['is_correct']]

    lt50 = [r for r in results if r['delay'] < 50 or r['delay'] == 0]
    correct_lt50 = sum(1 for r in lt50 if r['is_correct'])
    wrong_lt50 = [r for r in lt50 if not r['is_correct']]

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  전체: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  delay<50: {correct_lt50}/{len(lt50)} ({100*correct_lt50/len(lt50):.1f}%)" if lt50 else "")

    # 클래스별 정확도
    for cls in LABELS:
        cases = [r for r in results if r['true_label'] == cls]
        if not cases: continue
        cls_ok = sum(1 for r in cases if r['is_correct'])
        cls_wrong = [r for r in cases if not r['is_correct']]
        rts = [r['confirmed_sec'] - r['delay'] for r in cases
               if r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]
        rt_str = f"avg={np.mean(rts):.1f}s max={max(rts)}s" if rts else "N/A"
        status = "✅" if cls_ok == len(cases) else "❌"
        print(f"    {cls:>12s}: {cls_ok:>2d}/{len(cases):>2d} {status} | 반응: {rt_str}")
        for r in cls_wrong:
            print(f"      test{r['tid']}({r['true_label']}→{r['final']}, d={r['delay']})")

    # LOCA_CL 상세
    cl_all = [r for r in results if r['true_label'] == 'LOCA_CL']
    cl_lt50 = [r for r in cl_all if r['delay'] < 50]
    if cl_all:
        cl_ok = sum(1 for r in cl_all if r['is_correct'])
        cl_lt50_ok = sum(1 for r in cl_lt50 if r['is_correct'])
        print(f"\n  LOCA_CL 상세: 전체 {cl_ok}/{len(cl_all)}, d<50 {cl_lt50_ok}/{len(cl_lt50)}")


# ===== 실행 =====
print("\n" + "="*80)
print("  [2/2] 3가지 로직 비교 (201~300 데이터)")
print("="*80)

r1 = sim_lc(cache, answers, ALL_TIDS)
print_results("① 기존 LC 로직 (LOCA4연속 + Late Correction)", r1)

r2 = sim_no_lc(cache, answers, ALL_TIDS)
print_results("② LC 없는 기본 (4연속만)", r2)

r3 = sim_guard(cache, answers, ALL_TIDS, CL_N=4, GUARD_W=4, CL_THRESH=0.35)
print_results("③ 단순 가드 (CL=4, guard=4, thresh=0.35, LC없음)", r3)

# 최종 요약
print("\n" + "="*80)
print("  최종 비교 요약")
print("="*80)
for label, r in [("① LC 있음", r1), ("② LC 없음 (기본)", r2), ("③ 가드 (LC없음)", r3)]:
    total = len(r)
    correct = sum(1 for x in r if x['is_correct'])
    lt50 = [x for x in r if x['delay'] < 50 or x['delay'] == 0]
    c_lt50 = sum(1 for x in lt50 if x['is_correct'])
    wrong_tids = [f"t{x['tid']}" for x in r if not x['is_correct']]

    hl_rt = [x['confirmed_sec']-x['delay'] for x in r if x['true_label']=='LOCA_HL' and x['is_correct'] and x['confirmed_sec'] and x['delay']>0]
    cl_rt = [x['confirmed_sec']-x['delay'] for x in r if x['true_label']=='LOCA_CL' and x['is_correct'] and x['confirmed_sec'] and x['delay']>0]
    rcp_rt = [x['confirmed_sec']-x['delay'] for x in r if x['true_label']=='LOCA_RCP' and x['is_correct'] and x['confirmed_sec'] and x['delay']>0]

    hl_s = f"HL={np.mean(hl_rt):.1f}s" if hl_rt else "HL=N/A"
    cl_s = f"CL={np.mean(cl_rt):.1f}s" if cl_rt else "CL=N/A"
    rcp_s = f"RCP={np.mean(rcp_rt):.1f}s" if rcp_rt else "RCP=N/A"

    print(f"  {label:>20s}: {correct}/{total} (d<50: {c_lt50}/{len(lt50)}) | {hl_s} {cl_s} {rcp_s} | 오답: {','.join(wrong_tids) if wrong_tids else '없음'}")
print("="*80)
