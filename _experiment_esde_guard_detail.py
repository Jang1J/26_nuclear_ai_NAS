"""
ESDE 가드 상세 비교: 반응시간 + ESDE 케이스별 확정 시점 변화.
ESDE 2연속 vs 3연속의 반응시간 차이를 확인.
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


def simulate_detail(datasets, ESDE_CONFIRM=2):
    """반응시간 상세 비교."""
    GRACE_PERIOD = 3
    CL_N = 4
    HL_RCP_N = 4
    CL_GUARD_WINDOW = 5
    CL_GUARD_THRESH = 0.15
    SGTR_CONFIRM = 2

    results_by_class = {}
    esde_cases = []

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

            for i in range(len(preds)):
                sec = i + 1
                pred = preds[i]

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

            final = confirmed_label if confirmed_label else 'NORMAL'
            is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')

            # ESDE/SGTR 케이스 상세 기록
            if 'ESDE' in true_label or 'SGTR' in true_label:
                react = confirmed_sec - delay if confirmed_sec and delay > 0 else None
                esde_cases.append({
                    'ds': ds_name, 'tid': tid, 'true': true_label, 'pred': final,
                    'delay': delay, 'sec': confirmed_sec, 'react': react,
                    'ok': is_correct,
                })

            if is_correct and confirmed_sec and delay > 0:
                cls = true_label
                results_by_class.setdefault(cls, []).append(confirmed_sec - delay)

    return results_by_class, esde_cases


def main():
    datasets = load_all()

    print("=" * 80)
    print("  ESDE 2연속 vs 3연속: 반응시간 상세 비교 (300+dt5)")
    print("=" * 80)

    for esde_n in [2, 3]:
        print(f"\n{'='*60}")
        print(f"  ESDE_CONFIRM = {esde_n}")
        print(f"{'='*60}")

        by_class, esde_cases = simulate_detail(datasets, ESDE_CONFIRM=esde_n)

        # 클래스별 평균 반응시간
        print(f"\n  클래스별 평균 반응시간 (정답만):")
        for cls in LABELS[1:]:
            if cls in by_class and by_class[cls]:
                times = by_class[cls]
                print(f"    {cls:14s}: avg={np.mean(times):5.1f}s, "
                      f"median={np.median(times):5.1f}s, "
                      f"max={max(times):5.1f}s, n={len(times)}")

        # ESDE/SGTR 케이스 상세
        print(f"\n  ESDE/SGTR 케이스 상세:")
        for case in esde_cases:
            mark = 'O' if case['ok'] else 'X'
            sec_str = f"sec{case['sec']}" if case['sec'] else '미확정'
            react_str = f"react={case['react']:.0f}s" if case['react'] is not None else ''
            print(f"    [{case['ds']:7s}] test{case['tid']:>3d} "
                  f"{case['true']:14s} → {case['pred']:14s} {mark} "
                  f"d={case['delay']:>2d} {sec_str:>6s} {react_str}")

    # 변화 요약
    print(f"\n{'='*80}")
    print("  변화 요약: ESDE 2→3 연속")
    print(f"{'='*80}")
    _, cases_2 = simulate_detail(datasets, ESDE_CONFIRM=2)
    _, cases_3 = simulate_detail(datasets, ESDE_CONFIRM=3)

    for c2, c3 in zip(cases_2, cases_3):
        if c2['sec'] != c3['sec'] or c2['pred'] != c3['pred']:
            print(f"  [{c2['ds']:7s}] test{c2['tid']:>3d} {c2['true']:14s}: "
                  f"sec{c2['sec'] or '?':>3} → sec{c3['sec'] or '?':>3}, "
                  f"{c2['pred']:14s} → {c3['pred']:14s}")


if __name__ == '__main__':
    main()
