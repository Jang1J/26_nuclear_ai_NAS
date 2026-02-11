"""
ESDE 2연속 vs 3연속: 실제 점수 영향 계산.

대회 채점:
- 오답 = 0점
- 정답 = 1~10점 (반응시간 빠를수록, 확률 높을수록 높은 점수)

점수 공식 추정 (일반적 대회 기준):
  score = max(1, 10 - reaction_time * k) 또는
  score = f(reaction_time, probability)

정확한 공식이 없으므로, 다양한 가정으로 비교.
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
ESDE_INDICES = {7, 8}
SGTR_INDICES = {4, 5, 6}


def load_all():
    datasets = []
    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl', 'rb') as f:
        c200 = pickle.load(f)
    ans200 = {}
    with open('/Users/jangjaewon/Desktop/NAS/data/Real_test_data/answers.csv') as f:
        for row in csv.DictReader(f):
            ans200[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
    datasets.append(('200', c200, ans200))

    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_201_300.pkl', 'rb') as f:
        c300 = pickle.load(f)
    ans300 = {}
    with open('/Users/jangjaewon/Desktop/NAS/_test_data/test201_300/answers.csv') as f:
        for row in csv.DictReader(f):
            ans300[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
    datasets.append(('201-300', c300, ans300))

    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_dt5.pkl', 'rb') as f:
        cdt = pickle.load(f)
    ansdt = {}
    with open('/Users/jangjaewon/Desktop/NAS/_test_data/test_dt5_/answers.csv') as f:
        for row in csv.DictReader(f):
            ansdt[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
    datasets.append(('dt5', cdt, ansdt))

    return datasets


def simulate_with_scores(datasets, ESDE_CONFIRM=2):
    GRACE_PERIOD = 3
    CL_N = 4
    HL_RCP_N = 4
    CL_GUARD_WINDOW = 5
    CL_GUARD_THRESH = 0.15
    SGTR_CONFIRM = 2

    case_results = []

    for ds_name, cache, answers in datasets:
        for tid in sorted(answers.keys()):
            if tid not in cache:
                continue
            preds = cache[tid]['preds']
            probs = cache[tid]['probs']
            true_label = answers[tid]['label']
            delay = answers[tid].get('delay', 0)

            confirmed_label = None
            confirmed_sec = None
            confirmed_prob = None

            for i in range(len(preds)):
                sec = i + 1
                pred = preds[i]
                prob = probs[i]

                if confirmed_label is not None:
                    continue
                if pred == 0 or sec <= GRACE_PERIOD:
                    continue

                if pred in LOCA_INDICES:
                    if pred == CL_IDX:
                        n_req = CL_N
                    else:
                        n_req = HL_RCP_N
                        w = min(CL_GUARD_WINDOW, i + 1)
                        recent_cl = [probs[j][CL_IDX] for j in range(max(0, i+1-w), i+1)]
                        if max(recent_cl) > CL_GUARD_THRESH:
                            continue
                elif pred in ESDE_INDICES:
                    n_req = ESDE_CONFIRM
                elif pred in SGTR_INDICES:
                    n_req = SGTR_CONFIRM
                else:
                    n_req = 2

                if i + 1 >= n_req:
                    recent = preds[i+1-n_req:i+1]
                    if all(p == pred for p in recent):
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec
                        confirmed_prob = prob[pred]

            final = confirmed_label if confirmed_label else 'NORMAL'
            is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')
            react = confirmed_sec - delay if confirmed_sec and delay > 0 else None

            case_results.append({
                'ds': ds_name, 'tid': tid, 'true': true_label, 'pred': final,
                'ok': is_correct, 'delay': delay, 'sec': confirmed_sec,
                'react': react, 'prob': confirmed_prob,
            })

    return case_results


def score_formula_linear(react_time, prob, max_time=50):
    """선형 점수: 빠를수록 높은 점수. 정답이면 1~10."""
    if react_time is None:
        return 1.0  # NORMAL 맞추면 최소점
    # 반응시간 기반 (0초=10점, max_time초=1점)
    time_score = max(1, 10 - 9 * react_time / max_time)
    return round(time_score, 2)


def score_formula_step(react_time, prob):
    """구간별 점수 (더 현실적인 가정)."""
    if react_time is None:
        return 5.0
    if react_time <= 2:
        return 10
    elif react_time <= 5:
        return 8
    elif react_time <= 10:
        return 6
    elif react_time <= 20:
        return 4
    elif react_time <= 30:
        return 2
    else:
        return 1


def main():
    datasets = load_all()

    print("=" * 80)
    print("  ESDE 2연속 vs 3연속: 실제 점수 영향 분석 (300+dt5)")
    print("=" * 80)

    results_2 = simulate_with_scores(datasets, ESDE_CONFIRM=2)
    results_3 = simulate_with_scores(datasets, ESDE_CONFIRM=3)

    # ESDE 케이스만 뽑아서 비교
    print("\n[ESDE 케이스별 비교]")
    print(f"{'ds':>8s} test{'':2s} {'정답':>12s} | {'2연속':>6s} react prob  | {'3연속':>6s} react prob  | 변화")
    print("-" * 95)

    esde_score_diff_linear = 0
    esde_score_diff_step = 0
    esde_count = 0

    for r2, r3 in zip(results_2, results_3):
        if 'ESDE' not in r2['true']:
            continue
        esde_count += 1

        sec2 = r2['sec']
        sec3 = r3['sec']
        react2 = r2['react']
        react3 = r3['react']

        s2_lin = score_formula_linear(react2, r2['prob']) if r2['ok'] else 0
        s3_lin = score_formula_linear(react3, r3['prob']) if r3['ok'] else 0
        s2_step = score_formula_step(react2, r2['prob']) if r2['ok'] else 0
        s3_step = score_formula_step(react3, r3['prob']) if r3['ok'] else 0

        diff_lin = s3_lin - s2_lin
        diff_step = s3_step - s2_step

        esde_score_diff_linear += diff_lin
        esde_score_diff_step += diff_step

        if sec2 != sec3:
            r2_str = f"{react2}s" if react2 is not None else "-"
            r3_str = f"{react3}s" if react3 is not None else "-"
            print(f"  [{r2['ds']:7s}] {r2['tid']:>3d} {r2['true']:>12s} | "
                  f"sec{sec2:>2d} {r2_str:>5s} {r2['prob']:.3f}  | "
                  f"sec{sec3:>2d} {r3_str:>5s} {r3['prob']:.3f}  | "
                  f"선형:{diff_lin:+.2f} 구간:{diff_step:+.1f}")

    # CL150에서 새로 맞춘 7건의 점수
    # CL150은 delay=0이라 300 테스트에 없음. 대회에서만 가치 있음.
    # 300 테스트에서만 점수 계산
    print(f"\n\n[300+dt5 점수 총합 비교]")

    for name, formula in [("선형(0s=10,50s=1)", score_formula_linear),
                          ("구간(≤2s=10,≤5s=8,...)", score_formula_step)]:
        total_2 = 0
        total_3 = 0
        for r2, r3 in zip(results_2, results_3):
            s2 = formula(r2['react'], r2['prob']) if r2['ok'] else 0
            s3 = formula(r3['react'], r3['prob']) if r3['ok'] else 0
            total_2 += s2
            total_3 += s3

        print(f"\n  [{name}]")
        print(f"    2연속 총점: {total_2:.1f}")
        print(f"    3연속 총점: {total_3:.1f}")
        print(f"    차이: {total_3 - total_2:+.1f}")

    # ESDE만의 점수 차이
    print(f"\n\n[ESDE 케이스만 점수 차이]")
    print(f"  ESDE 케이스 수: {esde_count}개")
    print(f"  선형 점수 차이 합: {esde_score_diff_linear:+.2f}")
    print(f"  구간 점수 차이 합: {esde_score_diff_step:+.1f}")

    # worst case: 대회에서 ESDE가 많이 나오는 경우
    print(f"\n\n[시나리오 분석]")
    print(f"  대회 10문제 중 ESDE가 N개 있다고 가정:")
    for n_esde in [1, 2, 3, 4, 5]:
        # ESDE 평균 점수 손실
        avg_loss_per_esde = esde_score_diff_linear / esde_count if esde_count > 0 else 0
        total_loss = avg_loss_per_esde * n_esde
        # 대형 leak CL이 ESDE로 오진되면 0점
        # 대형 leak CL 나올 확률은 낮지만 1건이면 큰 손해
        gain_if_catch = 5  # 오진 방지 시 평균 점수 (보수적)
        print(f"  ESDE {n_esde}개: ESDE 점수 손실 = {total_loss:+.2f}, "
              f"대형leak CL 1건 오진 방지 이득 = +{gain_if_catch:.0f}")


if __name__ == '__main__':
    main()
