"""
Team 6 최종 로직 — 초별 로그 시뮬레이션.
각 테스트 케이스마다 sec 1~60 동안 무슨 일이 일어나는지 상세 출력.
"""
import pickle, csv
import numpy as np

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]

LOCA_INDICES = {1, 2, 3}
CL_IDX = 2
ESDE_INDICES = {7, 8}
SGTR_INDICES = {4, 5, 6}

GRACE_PERIOD = 3
CONFIRM_COUNT = 2
LOCA_CONFIRM_COUNT = 4
CL_GUARD_WINDOW = 5
CL_GUARD_THRESH = 0.15
ESDE_GUARD_WINDOW = 3
ESDE_GUARD_THRESH = 0.05


def run_demo(preds, probs, true_label, test_name, delay=0):
    """초별 상세 로그 출력."""
    print(f"\n{'='*90}")
    print(f"  {test_name} | 정답: {true_label} | delay: {delay}")
    print(f"{'='*90}")
    print(f"  {'sec':>3s} | {'예측':>12s} | {'확률':>6s} | {'연속':>4s} | {'가드체크':>30s} | {'상태'}")
    print(f"  {'-'*84}")

    confirmed_label = None
    confirmed_sec = None
    pred_history = []
    prob_history = []

    for i in range(len(preds)):
        sec = i + 1
        pred = preds[i]
        prob = probs[i]
        pred_label = LABELS[pred]
        pred_prob = prob[pred]

        pred_history.append(pred)
        prob_history.append(prob)

        # 연속 횟수 계산
        streak = 0
        for j in range(len(pred_history)-1, -1, -1):
            if pred_history[j] == pred:
                streak += 1
            else:
                break

        status = ""
        guard_info = ""

        if confirmed_label is not None:
            status = f"확정완료 ({confirmed_label})"
        elif pred == 0:
            status = "NORMAL 예측 → 무시"
        elif sec <= GRACE_PERIOD:
            status = f"Grace Period ({sec}/{GRACE_PERIOD})"
        else:
            # 확정 로직 체크
            if pred in LOCA_INDICES:
                n_req = LOCA_CONFIRM_COUNT
                if streak >= n_req:
                    if pred != CL_IDX:
                        # CL Guard
                        w = min(CL_GUARD_WINDOW, len(prob_history))
                        cl_probs = [prob_history[-j-1][CL_IDX] for j in range(w)]
                        max_cl = max(cl_probs)
                        guard_info = f"CL가드: max(CL)={max_cl:.3f} vs {CL_GUARD_THRESH}"
                        if max_cl > CL_GUARD_THRESH:
                            status = f"4연속 BUT CL가드 보류!"
                        else:
                            confirmed_label = LABELS[pred]
                            confirmed_sec = sec
                            status = f"*** 확정! {confirmed_label} ***"
                    else:
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec
                        status = f"*** 확정! {confirmed_label} (CL 4연속) ***"
                else:
                    status = f"LOCA {streak}/{n_req}연속 대기"

            elif pred in ESDE_INDICES:
                n_req = CONFIRM_COUNT
                if streak >= n_req:
                    # ESDE Guard
                    w = min(ESDE_GUARD_WINDOW, len(prob_history))
                    loca_max = 0
                    for j in range(1, w + 1):
                        loca_sum = sum(prob_history[-j][k] for k in LOCA_INDICES)
                        loca_max = max(loca_max, loca_sum)
                    guard_info = f"ESDE가드: max(LOCA)={loca_max:.3f} vs {ESDE_GUARD_THRESH}"
                    if loca_max > ESDE_GUARD_THRESH:
                        status = f"2연속 BUT ESDE가드 보류!"
                    else:
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec
                        status = f"*** 확정! {confirmed_label} ***"
                else:
                    status = f"ESDE {streak}/{n_req}연속 대기"

            elif pred in SGTR_INDICES:
                n_req = CONFIRM_COUNT
                if streak >= n_req:
                    confirmed_label = LABELS[pred]
                    confirmed_sec = sec
                    status = f"*** 확정! {confirmed_label} ***"
                else:
                    status = f"SGTR {streak}/{n_req}연속 대기"
            else:
                status = f"{streak}연속 대기"

        # top3 확률 표시
        top3_idx = np.argsort(prob)[::-1][:3]
        top3_str = " ".join(f"{LABELS[k][:6]}={prob[k]:.3f}" for k in top3_idx)

        print(f"  {sec:>3d} | {pred_label:>12s} | {pred_prob:>5.3f} | {streak:>4d} | {guard_info:>30s} | {status}")

        # 확정 직후 1줄 더 출력하고 중단 (나머지는 동일하니)
        if confirmed_label and sec == confirmed_sec:
            react = confirmed_sec - delay if delay > 0 else None
            react_str = f", react={react}s" if react is not None else ""
            ok = "O" if confirmed_label == true_label else "X"
            print(f"  {'':>3s}   → 결과: {ok} (정답={true_label}, 예측={confirmed_label}, sec={confirmed_sec}{react_str})")
            # 나머지 초는 생략
            remaining = len(preds) - sec
            if remaining > 0:
                print(f"  ... (이후 {remaining}초는 확정 완료 상태로 동일 전송)")
            break

    if confirmed_label is None:
        print(f"\n  → 60초 내 미확정 → NORMAL 처리 (정답={true_label}) {'O' if true_label == 'NORMAL' else 'X'}")


# ===== 데이터 로드 =====
# 대표 케이스 선정: 각 사고 유형별 1개 + 문제 케이스들
demo_cases = []

# 300 테스트에서 대표 케이스
with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl', 'rb') as f:
    c200 = pickle.load(f)
ans200 = {}
with open('/Users/jangjaewon/Desktop/NAS/data/Real_test_data/answers.csv') as f:
    for row in csv.DictReader(f):
        ans200[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}

# 대표 케이스 (각 유형별 1개씩)
# SGTR: 빠르게 확정되는 케이스
for tid in [1]:  # SGTR_Loop1, d=38
    demo_cases.append(('test1-200', tid, c200[tid], ans200[tid]))

# ESDE: 정상 확정 (가드 안 걸림)
for tid in [50]:  # ESDE_in, d=6
    demo_cases.append(('test1-200', tid, c200[tid], ans200[tid]))

# LOCA_HL: CL 가드 통과
for tid in [5]:  # LOCA_HL
    if 5 in c200 and 5 in ans200:
        demo_cases.append(('test1-200', tid, c200[tid], ans200[tid]))

# LOCA_CL: 4연속 확정
for tid in [57]:  # LOCA_CL
    if 57 in c200 and 57 in ans200:
        demo_cases.append(('test1-200', tid, c200[tid], ans200[tid]))

# 오답 케이스: test133 (CL→RCP)
for tid in [133]:
    demo_cases.append(('test1-200', tid, c200[tid], ans200[tid]))

# CL150에서 ESDE 가드가 작동하는 케이스
with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_cl150.pkl', 'rb') as f:
    ccl = pickle.load(f)
anscl = {}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/LOCA_CL_test/answers.csv') as f:
    for row in csv.DictReader(f):
        anscl[int(row['test_id'])] = {
            'label': row['label'], 'delay': 0,
            'leak_size': int(row['leak_size']), 'node': int(row['node']),
        }

# ESDE 가드 작동 케이스 (대형 leak CL → ESDE_in 오진 방지)
for tid in [88, 124]:  # leak=1120 n=8, leak=700 n=14
    demo_cases.append(('CL150', tid, ccl[tid], anscl[tid]))

# ESDE 가드 없이도 정답인 일반 CL 케이스
for tid in [5]:  # leak=100, n=1
    demo_cases.append(('CL150', tid, ccl[tid], anscl[tid]))

# ===== 실행 =====
print("=" * 90)
print("  Team 6 최종 로직 — 초별 시뮬레이션 데모")
print("  CL Guard: GW=5, T=0.15 | ESDE Guard: GW=3, T=0.05 | Grace=3")
print("=" * 90)

for ds_name, tid, cache_data, ans_data in demo_cases:
    preds = cache_data['preds']
    probs = cache_data['probs']
    true_label = ans_data['label']
    delay = ans_data.get('delay', 0)

    extra = ""
    if 'leak_size' in ans_data:
        extra = f" (leak={ans_data['leak_size']}, node={ans_data['node']})"

    test_name = f"[{ds_name}] test{tid}{extra}"
    run_demo(preds, probs, true_label, test_name, delay)
