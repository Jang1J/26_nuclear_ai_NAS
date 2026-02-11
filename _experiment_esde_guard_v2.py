"""
ESDE 가드 (2연속 유지 + LOCA확률 가드) 반응시간 상세 비교.
진짜 ESDE에는 영향 없이, 대형 leak CL→ESDE만 차단하는지 확인.
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

    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_cl150.pkl', 'rb') as f:
        ccl = pickle.load(f)
    anscl = {}
    with open('/Users/jangjaewon/Desktop/NAS/_test_data/LOCA_CL_test/answers.csv') as f:
        for row in csv.DictReader(f):
            anscl[int(row['test_id'])] = {
                'label': row['label'], 'delay': 0,
                'leak_size': int(row['leak_size']), 'node': int(row['node']),
            }
    datasets.append(('CL150', ccl, anscl))

    with open('/Users/jangjaewon/Desktop/NAS/_pred_cache_dt5.pkl', 'rb') as f:
        cdt = pickle.load(f)
    ansdt = {}
    with open('/Users/jangjaewon/Desktop/NAS/_test_data/test_dt5_/answers.csv') as f:
        for row in csv.DictReader(f):
            ansdt[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}
    datasets.append(('dt5', cdt, ansdt))

    return datasets


def simulate(datasets, ESDE_GUARD_WINDOW=0, ESDE_LOCA_THRESH=0.0):
    """ESDE 2연속 유지 + LOCA 확률 가드."""
    GRACE_PERIOD = 3
    CL_N = 4
    HL_RCP_N = 4
    CL_GUARD_WINDOW = 5
    CL_GUARD_THRESH = 0.15
    ESDE_CONFIRM = 2
    SGTR_CONFIRM = 2

    all_cases = []

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
                    # ESDE LOCA 가드
                    if ESDE_GUARD_WINDOW > 0:
                        w = min(ESDE_GUARD_WINDOW, i + 1)
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

                if i + 1 >= n_req:
                    recent = preds[i+1-n_req:i+1]
                    if all(p == pred for p in recent):
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec

            final = confirmed_label if confirmed_label else 'NORMAL'
            is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')
            react = confirmed_sec - delay if confirmed_sec and delay > 0 else None

            all_cases.append({
                'ds': ds_name, 'tid': tid, 'true': true_label, 'pred': final,
                'ok': is_correct, 'sec': confirmed_sec, 'react': react,
            })

    return all_cases


def main():
    datasets = load_all()

    print("=" * 80)
    print("  ESDE 가드 (2연속 유지 + LOCA확률 가드) vs 기준선 vs 3연속")
    print("=" * 80)

    # 기준선 (가드 없음)
    baseline = simulate(datasets, ESDE_GUARD_WINDOW=0)
    # 3연속 (비교용) — 별도 시뮬 필요. 여기선 가드 방식만 비교
    # ESDE 가드 GW=3, T=0.05
    guarded = simulate(datasets, ESDE_GUARD_WINDOW=3, ESDE_LOCA_THRESH=0.05)

    # ESDE 케이스 반응시간 비교
    print("\n[ESDE 케이스 반응시간 비교: 기준선 vs 가드(GW=3,T=0.05)]")
    print(f"{'':>8s} {'test':>4s} {'정답':>12s} | {'기준선':>6s} react | {'가드':>6s} react | 변화")
    print("-" * 80)

    changed_count = 0
    esde_baseline_reacts = []
    esde_guarded_reacts = []

    for b, g in zip(baseline, guarded):
        if 'ESDE' not in b['true'] and 'ESDE' not in b['pred']:
            continue

        b_react = b['react']
        g_react = g['react']

        if b['ok'] and b_react is not None:
            esde_baseline_reacts.append(b_react)
        if g['ok'] and g_react is not None:
            esde_guarded_reacts.append(g_react)

        if b['sec'] != g['sec'] or b['pred'] != g['pred']:
            changed_count += 1
            b_r = f"{b_react}s" if b_react is not None else "-"
            g_r = f"{g_react}s" if g_react is not None else "-"
            mark_b = 'O' if b['ok'] else 'X'
            mark_g = 'O' if g['ok'] else 'X'
            print(f"  [{b['ds']:7s}] {b['tid']:>3d} {b['true']:>12s} | "
                  f"sec{b['sec'] or '?':>2} {b_r:>5s} {mark_b} | "
                  f"sec{g['sec'] or '?':>2} {g_r:>5s} {mark_g} | "
                  f"{b['pred']:>12s} → {g['pred']:>12s}")

    print(f"\n  변화된 ESDE 케이스: {changed_count}개")
    if esde_baseline_reacts:
        print(f"  기준선 ESDE avg react: {np.mean(esde_baseline_reacts):.1f}s")
    if esde_guarded_reacts:
        print(f"  가드   ESDE avg react: {np.mean(esde_guarded_reacts):.1f}s")

    # 전체 정답 수
    print(f"\n[전체 정답 수]")
    for name, cases in [("기준선(가드없음)", baseline), ("가드(GW=3,T=0.05)", guarded)]:
        by_ds = {}
        for c in cases:
            ds = c['ds']
            by_ds.setdefault(ds, {'correct': 0, 'total': 0, 'wrong': []})
            by_ds[ds]['total'] += 1
            if c['ok']:
                by_ds[ds]['correct'] += 1
            else:
                by_ds[ds]['wrong'].append(f"test{c['tid']}({c['true']}→{c['pred']})")

        total_c = sum(v['correct'] for v in by_ds.values())
        total_t = sum(v['total'] for v in by_ds.values())
        parts = []
        for ds in ['200', '201-300', 'CL150', 'dt5']:
            if ds in by_ds:
                v = by_ds[ds]
                parts.append(f"{ds}:{v['correct']}/{v['total']}")
        print(f"  {name:30s}: {' | '.join(parts)} | 합계:{total_c}/{total_t}")
        for ds in by_ds:
            if by_ds[ds]['wrong']:
                print(f"    [{ds} 오답] {by_ds[ds]['wrong']}")

    # 여러 ESDE 가드 파라미터로 진짜 ESDE 반응시간 변화 확인
    print(f"\n\n[ESDE 가드 파라미터별 진짜 ESDE 반응시간]")
    print(f"{'파라미터':>25s} | {'ESDE avg':>8s} | {'변화':>6s} | {'변화된 ESDE':>10s} | 합계")
    print("-" * 80)

    baseline_esde_avg = np.mean(esde_baseline_reacts) if esde_baseline_reacts else 0

    for gw in [2, 3, 4, 5]:
        for thresh in [0.01, 0.03, 0.05, 0.10, 0.15]:
            cases = simulate(datasets, ESDE_GUARD_WINDOW=gw, ESDE_LOCA_THRESH=thresh)

            esde_reacts = []
            esde_changed = 0
            total_correct = sum(1 for c in cases if c['ok'])
            total = len(cases)

            for c, b in zip(cases, baseline):
                if 'ESDE' in c['true'] and c['ok'] and c['react'] is not None:
                    esde_reacts.append(c['react'])
                if c['sec'] != b['sec'] or c['pred'] != b['pred']:
                    if 'ESDE' in c['true']:
                        esde_changed += 1

            avg = np.mean(esde_reacts) if esde_reacts else 0
            diff = avg - baseline_esde_avg
            params = f"GW={gw}, T={thresh:.2f}"
            print(f"  {params:>23s} | {avg:>7.2f}s | {diff:>+5.2f}s | {esde_changed:>10d} | {total_correct}/{total}")


if __name__ == '__main__':
    main()
