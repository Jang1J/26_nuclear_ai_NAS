"""
ESDE 가드 파라미터 탐색.
대형 leak CL이 ESDE_in으로 오진되는 7건 개선 시도.

아이디어: ESDE_in/out 2연속 확정 시, 최근 N초간 LOCA 확률이 높으면 보류.
→ 확정 요건을 높이거나, LOCA prob 가드를 적용.

4개 캐시 (200+100+150+17 = 467개) 통합 테스트.
"""
import pickle, csv
from pathlib import Path
import numpy as np

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]

LOCA_INDICES = {1, 2, 3}
CL_IDX = 2
ESDE_IN_IDX = 7
ESDE_OUT_IDX = 8
ESDE_INDICES = {7, 8}
SGTR_INDICES = {4, 5, 6}

# ===== 캐시 + 정답 로드 =====
def load_all():
    """4개 캐시 + 정답 통합 로드."""
    datasets = []

    # 1) test 1~200
    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl', 'rb') as f:
        c200 = pickle.load(f)
    ans200 = {}
    with open('/Users/jangjaewon/Desktop/NAS/data/Real_test_data/answers.csv') as f:
        for row in csv.DictReader(f):
            ans200[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
    datasets.append(('200', c200, ans200))

    # 2) test 201~300
    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_201_300.pkl', 'rb') as f:
        c300 = pickle.load(f)
    ans300 = {}
    with open('/Users/jangjaewon/Desktop/NAS/_test_data/test201_300/answers.csv') as f:
        for row in csv.DictReader(f):
            ans300[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
    datasets.append(('201-300', c300, ans300))

    # 3) LOCA_CL 150
    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_cl150.pkl', 'rb') as f:
        ccl = pickle.load(f)
    anscl = {}
    with open('/Users/jangjaewon/Desktop/NAS/_test_data/LOCA_CL_test/answers.csv') as f:
        for row in csv.DictReader(f):
            anscl[int(row['test_id'])] = {
                'label': row['label'],
                'delay': 0,
                'leak_size': int(row['leak_size']),
                'node': int(row['node']),
            }
    datasets.append(('CL150', ccl, anscl))

    # 4) dt5 17
    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_dt5.pkl', 'rb') as f:
        cdt = pickle.load(f)
    ansdt = {}
    with open('/Users/jangjaewon/Desktop/NAS/_test_data/test_dt5_/answers.csv') as f:
        for row in csv.DictReader(f):
            ansdt[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
    datasets.append(('dt5', cdt, ansdt))

    return datasets


def simulate(datasets, CL_N=4, CL_GUARD_WINDOW=5, CL_GUARD_THRESH=0.15,
             ESDE_CONFIRM=2, ESDE_GUARD_WINDOW=0, ESDE_LOCA_THRESH=0.0,
             verbose=False):
    """
    확정 로직 시뮬레이션.

    ESDE 가드 옵션:
    - ESDE_CONFIRM: ESDE 확정에 필요한 연속 수 (기본 2)
    - ESDE_GUARD_WINDOW: ESDE 확정 시 LOCA 확률 확인 윈도우 (0=비활성)
    - ESDE_LOCA_THRESH: LOCA 확률 합 임계값 (이상이면 보류)
    """
    GRACE_PERIOD = 3
    HL_RCP_N = 4
    SGTR_CONFIRM = 2

    results = {}  # dataset_name -> {correct, wrong, wrong_list, react_times}

    for ds_name, cache, answers in datasets:
        correct = 0
        wrong = 0
        wrong_list = []
        react_times_by_class = {}

        for tid in sorted(answers.keys()):
            if tid not in cache:
                continue

            preds = cache[tid]['preds']
            probs = cache[tid]['probs']
            true_label = answers[tid]['label']
            delay = answers[tid].get('delay', 0)

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

                # 확정 요건 결정
                if pred in LOCA_INDICES:
                    if pred == CL_IDX:
                        n_req = CL_N
                    else:
                        n_req = HL_RCP_N
                        # CL 전환 가드
                        w = min(CL_GUARD_WINDOW, i + 1)
                        recent_cl = [probs[j][CL_IDX] for j in range(max(0, i+1-w), i+1)]
                        if max(recent_cl) > CL_GUARD_THRESH:
                            continue

                elif pred in ESDE_INDICES:
                    n_req = ESDE_CONFIRM

                    # ESDE 가드: LOCA 확률 확인
                    if ESDE_GUARD_WINDOW > 0:
                        w = min(ESDE_GUARD_WINDOW, i + 1)
                        # 최근 N초간 LOCA 확률 합(HL+CL+RCP)의 최대값
                        recent_loca_max = 0
                        for j in range(max(0, i+1-w), i+1):
                            loca_sum = sum(probs[j][k] for k in LOCA_INDICES)
                            recent_loca_max = max(recent_loca_max, loca_sum)
                        if recent_loca_max > ESDE_LOCA_THRESH:
                            continue

                elif pred in SGTR_INDICES:
                    n_req = SGTR_CONFIRM
                else:
                    n_req = 2

                # 연속 확인
                if i + 1 >= n_req:
                    recent = preds[i+1-n_req:i+1]
                    if all(p == pred for p in recent):
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec

            final = confirmed_label if confirmed_label else 'NORMAL'

            if final == true_label:
                correct += 1
                if confirmed_sec and delay > 0:
                    rt = confirmed_sec - delay
                    cls = true_label.split('_')[0] if '_' in true_label else true_label
                    if cls == 'LOCA':
                        cls = true_label  # LOCA_HL, LOCA_CL, LOCA_RCP 구분
                    react_times_by_class.setdefault(cls, []).append(rt)
            elif confirmed_label is None and true_label == 'NORMAL':
                correct += 1
            else:
                wrong += 1
                info = f"test{tid}({true_label}"
                if 'leak_size' in answers[tid]:
                    info += f",leak={answers[tid]['leak_size']},n={answers[tid]['node']}"
                elif delay > 0:
                    info += f",d={delay}"
                info += f"→{final})"
                wrong_list.append(info)

        results[ds_name] = {
            'correct': correct,
            'total': len([t for t in answers if t in cache]),
            'wrong': wrong,
            'wrong_list': wrong_list,
            'react_times': react_times_by_class,
        }

    return results


def print_results(results, params_str=""):
    total_correct = sum(r['correct'] for r in results.values())
    total_all = sum(r['total'] for r in results.values())

    # 300 d<50 계산
    d50_correct = 0
    d50_total = 0
    for ds_name in ['200', '201-300']:
        if ds_name in results:
            r = results[ds_name]
            d50_correct += r['correct']
            d50_total += r['total']
            # delay>50인 오답은 d<50에서 제외해야 하지만, 간단히 300 전체로 표시

    line = f"{params_str:40s} | 300: {d50_correct}/{d50_total}"
    if 'CL150' in results:
        r = results['CL150']
        line += f" | CL150: {r['correct']}/{r['total']}"
    if 'dt5' in results:
        r = results['dt5']
        line += f" | dt5: {r['correct']}/{r['total']}"
    line += f" | 합계: {total_correct}/{total_all}"
    print(line)


def main():
    print("=" * 80)
    print("  ESDE 가드 파라미터 탐색")
    print("  기존 최적: GW=5, T=0.15 (ESDE 가드 없음)")
    print("=" * 80)

    datasets = load_all()
    print(f"  데이터셋 로드 완료")
    for ds_name, cache, answers in datasets:
        print(f"  {ds_name}: {len(cache)}개 캐시, {len(answers)}개 정답")
    print()

    # ===== 기준선: 현재 로직 (ESDE 가드 없음) =====
    print("=" * 80)
    print("[기준선] 현재 로직 (ESDE 가드 없음, ESDE 2연속)")
    print("=" * 80)
    baseline = simulate(datasets, CL_N=4, CL_GUARD_WINDOW=5, CL_GUARD_THRESH=0.15,
                        ESDE_CONFIRM=2, ESDE_GUARD_WINDOW=0)
    print_results(baseline, "기준선 (ESDE_CONFIRM=2, 가드=없음)")
    for ds, r in baseline.items():
        if r['wrong_list']:
            print(f"  [{ds} 오답] {r['wrong_list']}")
    print()

    # ===== 방법 1: ESDE 확정 연속수 증가 =====
    print("=" * 80)
    print("[방법 1] ESDE 확정 연속수 증가 (가드 없음)")
    print("=" * 80)
    for esde_n in [3, 4, 5]:
        r = simulate(datasets, CL_N=4, CL_GUARD_WINDOW=5, CL_GUARD_THRESH=0.15,
                     ESDE_CONFIRM=esde_n, ESDE_GUARD_WINDOW=0)
        print_results(r, f"ESDE_CONFIRM={esde_n}")
        # ESDE 관련 오답만 표시
        for ds, rr in r.items():
            esde_wrongs = [w for w in rr['wrong_list'] if 'ESDE' in w or ('LOCA_CL' in w and 'ESDE' in w)]
            other_wrongs = [w for w in rr['wrong_list'] if w not in esde_wrongs]
            if esde_wrongs:
                print(f"    [{ds} ESDE관련] {esde_wrongs}")
            if other_wrongs:
                print(f"    [{ds} 기타오답] {other_wrongs}")
    print()

    # ===== 방법 2: ESDE LOCA 확률 가드 =====
    print("=" * 80)
    print("[방법 2] ESDE LOCA 확률 가드 (ESDE 2연속 유지)")
    print("=" * 80)
    for gw in [3, 4, 5]:
        for thresh in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            r = simulate(datasets, CL_N=4, CL_GUARD_WINDOW=5, CL_GUARD_THRESH=0.15,
                         ESDE_CONFIRM=2, ESDE_GUARD_WINDOW=gw, ESDE_LOCA_THRESH=thresh)
            params = f"ESDE_GW={gw}, LOCA_T={thresh:.2f}"
            print_results(r, params)
    print()

    # ===== 방법 3: ESDE 연속 증가 + LOCA 확률 가드 조합 =====
    print("=" * 80)
    print("[방법 3] ESDE 연속 증가 + LOCA 확률 가드 조합")
    print("=" * 80)
    for esde_n in [3, 4]:
        for gw in [3, 5]:
            for thresh in [0.10, 0.20, 0.30]:
                r = simulate(datasets, CL_N=4, CL_GUARD_WINDOW=5, CL_GUARD_THRESH=0.15,
                             ESDE_CONFIRM=esde_n, ESDE_GUARD_WINDOW=gw, ESDE_LOCA_THRESH=thresh)
                params = f"ESDE_N={esde_n}, GW={gw}, T={thresh:.2f}"
                print_results(r, params)
    print()

    # ===== 최적 찾기 =====
    print("=" * 80)
    print("[전수 탐색] 최적 파라미터 찾기")
    print("=" * 80)

    best_total = 0
    best_params = None
    best_results = None

    for esde_n in [2, 3, 4, 5]:
        for gw in [0, 2, 3, 4, 5, 6]:
            for thresh in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                if gw == 0 and thresh > 0:
                    continue
                if gw > 0 and thresh == 0:
                    continue
                r = simulate(datasets, CL_N=4, CL_GUARD_WINDOW=5, CL_GUARD_THRESH=0.15,
                             ESDE_CONFIRM=esde_n, ESDE_GUARD_WINDOW=gw, ESDE_LOCA_THRESH=thresh)
                total_correct = sum(rr['correct'] for rr in r.values())
                total_all = sum(rr['total'] for rr in r.values())
                if total_correct > best_total:
                    best_total = total_correct
                    best_params = (esde_n, gw, thresh)
                    best_results = {ds: dict(rr) for ds, rr in r.items()}

    print(f"\n최적 파라미터: ESDE_N={best_params[0]}, GW={best_params[1]}, T={best_params[2]:.2f}")
    print(f"총점: {best_total}/{sum(rr['total'] for rr in best_results.values())}")
    print_results(best_results, f"최적 ESDE_N={best_params[0]}, GW={best_params[1]}, T={best_params[2]:.2f}")
    for ds, r in best_results.items():
        if r['wrong_list']:
            print(f"  [{ds} 오답] {r['wrong_list']}")

    # 기준선과 비교
    baseline_total = sum(r['correct'] for r in baseline.values())
    print(f"\n개선: {baseline_total} → {best_total} (+{best_total - baseline_total})")


if __name__ == '__main__':
    main()
