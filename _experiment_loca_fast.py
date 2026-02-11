"""
LOCA 확정 횟수 줄이기 실험.
현재: 무조건 4연속 → 제안: 확률 높으면 더 빨리 확정.
467개 전체 테스트로 시뮬레이션.

실험 조합:
  A) LOCA 3연속 (확률 가드 없이)
  B) LOCA 2연속 (확률 가드 없이)
  C) LOCA: 확률 >= thresh → 1초 즉시, 아니면 4연속
  D) LOCA: 확률 >= thresh → 2연속, 아니면 4연속
  E) LOCA: 확률 >= thresh → 3연속, 아니면 4연속
"""
import pickle, csv
import numpy as np

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]

GRACE_PERIOD = 3
CONFIRM_COUNT = 2
LOCA_INDICES = {1, 2, 3}
CL_IDX = 2
ESDE_INDICES = {7, 8}
SGTR_INDICES = {4, 5, 6}

CL_GUARD_WINDOW = 5
CL_GUARD_THRESH = 0.15
ESDE_GUARD_WINDOW = 3
ESDE_GUARD_THRESH = 0.05


def simulate(preds, probs, loca_n=4, loca_fast_thresh=None, loca_fast_n=1):
    """
    loca_n: 기본 LOCA 연속 확정 횟수 (기존 4)
    loca_fast_thresh: 확률 >= 이 값이면 loca_fast_n 연속으로 줄임
    loca_fast_n: fast 모드 연속 횟수 (1, 2, 3)
    """
    confirmed_label = None
    confirmed_sec = None

    for i in range(len(preds)):
        sec = i + 1
        pred = preds[i]
        prob = probs[i]

        if confirmed_label is not None:
            continue
        if pred == 0:
            continue
        if sec <= GRACE_PERIOD:
            continue

        hist_len = i + 1

        if pred in LOCA_INDICES:
            # 확률 높으면 fast_n, 아니면 기본 loca_n
            if loca_fast_thresh is not None and prob[pred] >= loca_fast_thresh:
                n_confirm = loca_fast_n
            else:
                n_confirm = loca_n

            if hist_len >= n_confirm:
                recent = preds[i+1-n_confirm:i+1]
                if all(p == pred for p in recent):
                    if pred != CL_IDX:
                        # CL Guard
                        w = min(CL_GUARD_WINDOW, hist_len)
                        recent_cl = [probs[j][CL_IDX] for j in range(max(0, i+1-w), i+1)]
                        if max(recent_cl) > CL_GUARD_THRESH:
                            pass
                        else:
                            confirmed_label = LABELS[pred]
                            confirmed_sec = sec
                    else:
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec

        elif pred in ESDE_INDICES:
            n_confirm = CONFIRM_COUNT
            if hist_len >= n_confirm:
                recent = preds[i+1-n_confirm:i+1]
                if all(p == pred for p in recent):
                    w = min(ESDE_GUARD_WINDOW, hist_len)
                    recent_loca_max = 0
                    for j in range(max(0, i+1-w), i+1):
                        loca_sum = sum(probs[j][k] for k in LOCA_INDICES)
                        recent_loca_max = max(recent_loca_max, loca_sum)
                    if recent_loca_max > ESDE_GUARD_THRESH:
                        pass
                    else:
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec
        else:
            # SGTR
            n_confirm = CONFIRM_COUNT
            if hist_len >= n_confirm:
                recent = preds[i+1-n_confirm:i+1]
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


# ===== 기준 결과 (현재 로직) =====
base_results = {}
for ds_name, cache, answers in datasets:
    for tid in sorted(answers.keys()):
        if tid not in cache:
            continue
        preds = cache[tid]['preds']
        probs_arr = cache[tid]['probs']
        final, conf_sec = simulate(preds, probs_arr, loca_n=4)
        base_results[(ds_name, tid)] = (final, conf_sec)


def run_experiment(label, loca_n, loca_fast_thresh, loca_fast_n):
    total_correct = 0
    total_all = 0
    wrong_list = []
    loca_reacts = {'HL': [], 'CL': [], 'RCP': []}
    new_wrong = []
    new_correct = []
    faster_cases = []

    for ds_name, cache, answers in datasets:
        for tid in sorted(answers.keys()):
            if tid not in cache:
                continue
            preds = cache[tid]['preds']
            probs_arr = cache[tid]['probs']
            true_label = answers[tid]['label']
            delay = answers[tid].get('delay', 0)

            final, conf_sec = simulate(preds, probs_arr,
                                       loca_n=loca_n,
                                       loca_fast_thresh=loca_fast_thresh,
                                       loca_fast_n=loca_fast_n)

            base_final, base_sec = base_results[(ds_name, tid)]
            is_correct = (final == true_label) or (final == 'NORMAL' and true_label == 'NORMAL')
            base_correct = (base_final == true_label) or (base_final == 'NORMAL' and true_label == 'NORMAL')

            total_all += 1
            if is_correct:
                total_correct += 1
            else:
                wrong_list.append(f"{ds_name}/test{tid}({true_label}→{final})")

            # 신규 오답/정답
            if not is_correct and base_correct:
                new_wrong.append(f"{ds_name}/test{tid}({true_label}→{final})")
            if is_correct and not base_correct:
                new_correct.append(f"{ds_name}/test{tid}({true_label})")

            # LOCA 반응시간
            if is_correct and conf_sec and delay > 0 and 'LOCA' in true_label:
                react = conf_sec - delay
                if 'HL' in true_label:
                    loca_reacts['HL'].append(react)
                elif 'CL' in true_label:
                    loca_reacts['CL'].append(react)
                elif 'RCP' in true_label:
                    loca_reacts['RCP'].append(react)

                if base_sec and conf_sec < base_sec:
                    faster_cases.append(
                        f"  {ds_name}/test{tid} {true_label}: {base_sec-delay}s→{react}s (-{base_sec-conf_sec}s)"
                    )

    hl_avg = f"{np.mean(loca_reacts['HL']):.2f}s" if loca_reacts['HL'] else "N/A"
    cl_avg = f"{np.mean(loca_reacts['CL']):.2f}s" if loca_reacts['CL'] else "N/A"
    rcp_avg = f"{np.mean(loca_reacts['RCP']):.2f}s" if loca_reacts['RCP'] else "N/A"

    print(f"\n{'─'*80}")
    print(f"  {label}")
    print(f"  정답: {total_correct}/{total_all} ({100*total_correct/total_all:.1f}%)")
    print(f"  HL avg: {hl_avg} ({len(loca_reacts['HL'])}건) | CL avg: {cl_avg} ({len(loca_reacts['CL'])}건) | RCP avg: {rcp_avg} ({len(loca_reacts['RCP'])}건)")
    print(f"  빨라진: {len(faster_cases)}건 | 신규오답: {len(new_wrong)}건 | 신규정답: {len(new_correct)}건")

    if new_wrong:
        print(f"  *** 신규 오답 ***:")
        for w in new_wrong:
            print(f"    {w}")
    if new_correct:
        print(f"  *** 신규 정답 ***:")
        for w in new_correct:
            print(f"    {w}")
    if faster_cases and len(faster_cases) <= 15:
        print(f"  빨라진 상세:")
        for fc in faster_cases:
            print(fc)
    elif faster_cases:
        print(f"  빨라진 상세 (상위 10건):")
        for fc in faster_cases[:10]:
            print(fc)
        print(f"  ... 외 {len(faster_cases)-10}건")


print("=" * 80)
print("  LOCA 확정 최적화 실험 (467개)")
print("  기준: LOCA 4연속 + CL Guard(GW=5, T=0.15)")
print("=" * 80)

# 기준
run_experiment("기존: LOCA 4연속", loca_n=4, loca_fast_thresh=None, loca_fast_n=1)

# A) 연속 횟수만 줄이기
run_experiment("LOCA 3연속 (가드 없이)", loca_n=3, loca_fast_thresh=None, loca_fast_n=1)
run_experiment("LOCA 2연속 (가드 없이)", loca_n=2, loca_fast_thresh=None, loca_fast_n=1)

# B) 확률 가드: 높으면 3연속
for t in [0.90, 0.95, 0.99]:
    run_experiment(f"확률>={t} → 3연속, else 4연속", loca_n=4, loca_fast_thresh=t, loca_fast_n=3)

# C) 확률 가드: 높으면 2연속
for t in [0.90, 0.95, 0.99]:
    run_experiment(f"확률>={t} → 2연속, else 4연속", loca_n=4, loca_fast_thresh=t, loca_fast_n=2)

# D) 확률 가드: 높으면 1초
for t in [0.95, 0.99, 0.999]:
    run_experiment(f"확률>={t} → 1초확정, else 4연속", loca_n=4, loca_fast_thresh=t, loca_fast_n=1)

print(f"\n{'='*80}")
