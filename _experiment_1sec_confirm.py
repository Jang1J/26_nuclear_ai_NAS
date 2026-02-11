"""
ESDE/SGTR 1초 확정 + 확률 가드 실험.
현재: 무조건 2연속 → 제안: 확률 높으면 1초 즉시 확정, 낮으면 2연속 대기.
467개 전체 테스트로 시뮬레이션.
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
LOCA_CONFIRM_COUNT = 4
LOCA_INDICES = {1, 2, 3}
CL_IDX = 2
ESDE_INDICES = {7, 8}
SGTR_INDICES = {4, 5, 6}

CL_GUARD_WINDOW = 5
CL_GUARD_THRESH = 0.15
ESDE_GUARD_WINDOW = 3
ESDE_GUARD_THRESH = 0.05


def simulate(preds, probs, fast_thresh=None):
    """
    fast_thresh: ESDE/SGTR 1초 즉시 확정 임계값.
      None이면 기존 로직 (2연속).
      값이 있으면: 확률 >= fast_thresh → 1초 즉시 확정, 아니면 2연속 대기.
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
            # LOCA: 기존 로직 그대로 (4연속 + CL Guard)
            n_confirm = LOCA_CONFIRM_COUNT
            if hist_len >= n_confirm:
                recent = preds[i+1-n_confirm:i+1]
                if all(p == pred for p in recent):
                    if pred != CL_IDX:
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
            if fast_thresh is not None and prob[pred] >= fast_thresh:
                # 확률 높으면 1초 즉시 확정 (ESDE Guard는 여전히 적용)
                w = min(ESDE_GUARD_WINDOW, hist_len)
                recent_loca_max = 0
                for j in range(max(0, i+1-w), i+1):
                    loca_sum = sum(probs[j][k] for k in LOCA_INDICES)
                    recent_loca_max = max(recent_loca_max, loca_sum)
                if recent_loca_max > ESDE_GUARD_THRESH:
                    pass  # ESDE Guard 보류
                else:
                    confirmed_label = LABELS[pred]
                    confirmed_sec = sec
            else:
                # 확률 낮으면 기존 2연속 + ESDE Guard
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

        elif pred in SGTR_INDICES:
            if fast_thresh is not None and prob[pred] >= fast_thresh:
                # 확률 높으면 1초 즉시 확정
                confirmed_label = LABELS[pred]
                confirmed_sec = sec
            else:
                # 확률 낮으면 기존 2연속
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


# ===== 시뮬레이션 =====
thresholds = [None, 0.90, 0.95, 0.99, 0.999]

print("=" * 100)
print("  ESDE/SGTR 1초 확정 + 확률 가드 실험")
print("  None = 기존(2연속), 숫자 = 확률 >= 이 값이면 1초 즉시 확정")
print("=" * 100)

for thresh in thresholds:
    total_correct = 0
    total_all = 0
    wrong_list = []
    esde_reacts = []
    sgtr_reacts = []
    faster_cases = []  # 기존보다 빨라진 케이스

    for ds_name, cache, answers in datasets:
        for tid in sorted(answers.keys()):
            if tid not in cache:
                continue
            preds = cache[tid]['preds']
            probs_arr = cache[tid]['probs']
            true_label = answers[tid]['label']
            delay = answers[tid].get('delay', 0)

            final, conf_sec = simulate(preds, probs_arr, fast_thresh=thresh)
            # 기존 로직 결과도 계산
            final_base, conf_sec_base = simulate(preds, probs_arr, fast_thresh=None)

            is_correct = (final == true_label) or (final == 'NORMAL' and true_label == 'NORMAL')
            total_all += 1

            if is_correct:
                total_correct += 1
            else:
                info = f"{ds_name}/test{tid}({true_label}→{final})"
                wrong_list.append(info)

            # 반응시간 비교
            if is_correct and conf_sec and delay > 0:
                react = conf_sec - delay
                if 'ESDE' in true_label:
                    esde_reacts.append(react)
                elif 'SGTR' in true_label:
                    sgtr_reacts.append(react)

                # 빨라진 케이스 추적
                if conf_sec_base and conf_sec < conf_sec_base:
                    faster_cases.append(
                        f"  {ds_name}/test{tid} {true_label}: {conf_sec_base-delay}s → {react}s (-{conf_sec_base-conf_sec}s)"
                    )

    thresh_str = f"thresh={thresh}" if thresh else "기존(2연속)"
    esde_avg = f"{np.mean(esde_reacts):.2f}s" if esde_reacts else "N/A"
    sgtr_avg = f"{np.mean(sgtr_reacts):.2f}s" if sgtr_reacts else "N/A"

    print(f"\n{'─'*80}")
    print(f"  {thresh_str}")
    print(f"  정답: {total_correct}/{total_all} ({100*total_correct/total_all:.1f}%)")
    print(f"  ESDE avg react: {esde_avg} ({len(esde_reacts)}건)")
    print(f"  SGTR avg react: {sgtr_avg} ({len(sgtr_reacts)}건)")
    print(f"  빨라진 케이스: {len(faster_cases)}건")

    if wrong_list:
        print(f"  오답:")
        for w in wrong_list:
            print(f"    {w}")

    if faster_cases and len(faster_cases) <= 20:
        print(f"  빨라진 상세:")
        for fc in faster_cases:
            print(fc)
    elif faster_cases:
        print(f"  빨라진 상세 (상위 10건):")
        for fc in faster_cases[:10]:
            print(fc)
        print(f"  ... 외 {len(faster_cases)-10}건")

print(f"\n{'='*100}")
