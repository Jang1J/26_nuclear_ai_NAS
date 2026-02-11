"""
GPT 제안: Per-class Risk Gating 로직 시뮬레이션
- SGTR/ESDE: 기존 2연속 빠른 확정
- LOCA_HL/RCP: 확률 기반 빠른 확정 OR CL 위험시 보류
- LOCA_CL: 보수적 확정 (p(CL)>=0.90 N연속)
- HOLD_MAX 타임아웃
- LC 완전 제거 (처음 보낸 게 최종)
"""
import csv, pickle
from pathlib import Path
from collections import Counter
import numpy as np

CACHE_FILE = '/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl'

answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}

with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

LABELS = ['NORMAL', 'LOCA_HL', 'LOCA_CL', 'LOCA_RCP',
          'SGTR_Loop1', 'SGTR_Loop2', 'SGTR_Loop3',
          'ESDE_in', 'ESDE_out']

LOCA_INDICES = {1, 2, 3}
CL_IDX = 2
ALL_TIDS = sorted(answers.keys())
GRACE = 3


def simulate_risk_gating(FAST_CONF, FAST_MARGIN, CL_DANGER_THRESH,
                          CL_CONFIRM_CONF, CL_CONFIRM_N, HOLD_MAX):
    """
    Risk Gating 로직:
    - HL/RCP 빠른 확정: p(top1) >= FAST_CONF, margin >= FAST_MARGIN, p(CL) <= CL_DANGER_THRESH
    - HL/RCP 보류: 위 조건 불만족 시 HOLD_MAX초까지 대기
    - CL 확정: p(CL) >= CL_CONFIRM_CONF가 CL_CONFIRM_N연속
    - 보류 타임아웃: HOLD_MAX 넘으면 그때 top1으로 확정
    - SGTR/ESDE: 기존 2연속
    """
    CONFIRM_SGTR_ESDE = 2
    results = []

    for tid in ALL_TIDS:
        if tid not in cache:
            continue
        preds = cache[tid]['preds']
        probs = cache[tid]['probs']

        confirmed_label = None
        confirmed_sec = None

        # 보류 상태
        hold_active = False
        hold_start = None
        hold_candidate = None  # 보류 시작 시 top1

        # 비정상 연속 카운트 (기존 로직 호환)
        consecutive_count = 0
        consecutive_label = None

        # CL 연속 카운트
        cl_high_streak = 0

        for i in range(len(preds)):
            sec = i + 1
            pred = preds[i]
            prob = probs[i]

            if confirmed_label is not None:
                continue

            if pred == 0 or sec <= GRACE:
                consecutive_count = 0
                consecutive_label = None
                cl_high_streak = 0
                hold_active = False
                hold_start = None
                continue

            # CL 고확률 연속 카운트
            if prob[CL_IDX] >= CL_CONFIRM_CONF:
                cl_high_streak += 1
            else:
                cl_high_streak = 0

            # 연속 카운트 업데이트
            if pred == (LABELS.index(consecutive_label) if consecutive_label else -1):
                consecutive_count += 1
            else:
                consecutive_count = 1
                consecutive_label = LABELS[pred]

            # ===== SGTR / ESDE: 빠른 확정 =====
            if pred not in LOCA_INDICES:
                if consecutive_count >= CONFIRM_SGTR_ESDE:
                    confirmed_label = LABELS[pred]
                    confirmed_sec = sec
                continue

            # ===== LOCA 계열 =====

            # CL 확정: p(CL) >= CL_CONFIRM_CONF가 CL_CONFIRM_N연속
            if pred == CL_IDX and cl_high_streak >= CL_CONFIRM_N:
                confirmed_label = LABELS[CL_IDX]
                confirmed_sec = sec
                continue

            # HL/RCP 확정 로직
            if pred in (1, 3):  # HL or RCP
                # 정렬된 확률로 마진 계산
                sorted_probs = sorted(prob, reverse=True)
                top1_prob = sorted_probs[0]
                second_prob = sorted_probs[1]
                margin = top1_prob - second_prob
                cl_prob = prob[CL_IDX]

                if not hold_active:
                    # 빠른 확정 조건 체크
                    if (top1_prob >= FAST_CONF and
                        margin >= FAST_MARGIN and
                        cl_prob <= CL_DANGER_THRESH and
                        consecutive_count >= 4):  # 최소 4연속은 유지
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec
                    elif consecutive_count >= 4:
                        # 4연속 달성했지만 빠른 확정 조건 불충분 → 보류 시작
                        hold_active = True
                        hold_start = sec
                        hold_candidate = LABELS[pred]
                else:
                    # 보류 중
                    hold_elapsed = sec - hold_start

                    # 보류 중 빠른 확정 조건 충족하면 즉시 확정
                    if (top1_prob >= FAST_CONF and
                        margin >= FAST_MARGIN and
                        cl_prob <= CL_DANGER_THRESH):
                        confirmed_label = LABELS[pred]
                        confirmed_sec = sec
                        hold_active = False
                    # 타임아웃
                    elif hold_elapsed >= HOLD_MAX:
                        confirmed_label = LABELS[pred]  # 현재 top1으로 확정
                        confirmed_sec = sec
                        hold_active = False

            # 보류 중에 CL이 top1이 되면 CL 확정 체크
            if hold_active and pred == CL_IDX and cl_high_streak >= CL_CONFIRM_N:
                confirmed_label = LABELS[CL_IDX]
                confirmed_sec = sec
                hold_active = False

        final = confirmed_label if confirmed_label else 'NORMAL'
        true_label = answers[tid]['label']
        is_correct = (final == true_label) or (confirmed_label is None and true_label == 'NORMAL')
        results.append({'tid': tid, 'true_label': true_label, 'delay': answers[tid]['delay'],
                        'final': final, 'confirmed_sec': confirmed_sec, 'is_correct': is_correct})
    return results


def print_stats(label, results):
    lt50 = [r for r in results if r['delay'] < 50 or r['delay'] == 0]
    correct_lt50 = sum(1 for r in lt50 if r['is_correct'])
    wrong_lt50 = [r for r in lt50 if not r['is_correct']]

    cl_lt50 = [r for r in results if r['true_label'] == 'LOCA_CL' and r['delay'] < 50]
    cl_lt50_ok = sum(1 for r in cl_lt50 if r['is_correct'])

    hl_rts = [r['confirmed_sec'] - r['delay'] for r in results
              if r['true_label'] == 'LOCA_HL' and r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]
    cl_rts = [r['confirmed_sec'] - r['delay'] for r in results
              if r['true_label'] == 'LOCA_CL' and r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]
    rcp_rts = [r['confirmed_sec'] - r['delay'] for r in results
               if r['true_label'] == 'LOCA_RCP' and r['is_correct'] and r['confirmed_sec'] and r['delay'] > 0]

    wrong_str = ", ".join([f"t{r['tid']}({r['true_label'][-2:]}→{r['final'][-2:] if r['final'] != 'NORMAL' else 'NOR'})" for r in wrong_lt50])
    if not wrong_str:
        wrong_str = "없음 ✅"

    hl_avg = np.mean(hl_rts) if hl_rts else 0
    cl_avg = np.mean(cl_rts) if cl_rts else 0
    rcp_avg = np.mean(rcp_rts) if rcp_rts else 0
    hl_max = max(hl_rts) if hl_rts else 0
    cl_max = max(cl_rts) if cl_rts else 0
    rcp_max = max(rcp_rts) if rcp_rts else 0

    return {
        'label': label,
        'correct_lt50': correct_lt50, 'total_lt50': len(lt50),
        'cl_ok': cl_lt50_ok, 'cl_total': len(cl_lt50),
        'hl_avg': hl_avg, 'hl_max': hl_max,
        'cl_avg': cl_avg, 'cl_max': cl_max,
        'rcp_avg': rcp_avg, 'rcp_max': rcp_max,
        'wrong_str': wrong_str,
    }


# 파라미터 탐색
print("=" * 140)
print("  Risk Gating 파라미터 탐색 (LC 없음)")
print("  기준: 기존 LC 로직 → HL=3.1s, CL=4.9s, RCP=3.3s (187/187)")
print("=" * 140)
print(f"{'파라미터':>55s} | d<50    | CL    | HL avg/max | CL avg/max | RCP avg/max | 틀린것")
print("-" * 140)

good_results = []

for FAST_CONF in [0.85, 0.90, 0.95]:
    for FAST_MARGIN in [0.15, 0.20, 0.30]:
        for CL_DANGER in [0.10, 0.15, 0.20, 0.30, 0.40]:
            for CL_CONF in [0.80, 0.85, 0.90]:
                for CL_N in [2, 3]:
                    for HOLD_MAX in [3, 4, 5, 6]:
                        results = simulate_risk_gating(
                            FAST_CONF, FAST_MARGIN, CL_DANGER,
                            CL_CONF, CL_N, HOLD_MAX)

                        lt50 = [r for r in results if r['delay'] < 50 or r['delay'] == 0]
                        correct_lt50 = sum(1 for r in lt50 if r['is_correct'])

                        if correct_lt50 < 185:
                            continue

                        s = print_stats("", results)
                        param = f"conf={FAST_CONF} margin={FAST_MARGIN} clD={CL_DANGER} clC={CL_CONF} clN={CL_N} hold={HOLD_MAX}"

                        print(f"  {param:>55s} | {s['correct_lt50']:>3d}/{s['total_lt50']} | {s['cl_ok']:>2d}/{s['cl_total']:>2d} | {s['hl_avg']:>4.1f}/{s['hl_max']:>2d}s   | {s['cl_avg']:>4.1f}/{s['cl_max']:>2d}s   | {s['rcp_avg']:>4.1f}/{s['rcp_max']:>2d}s    | {s['wrong_str']}")

                        good_results.append({
                            'param': param, **s,
                            'total_loca_avg': s['hl_avg'] + s['cl_avg'] + s['rcp_avg'],
                        })

# 최적 결과 정리
if good_results:
    print("\n" + "=" * 140)

    # 187/187 달성하는 것들
    perfect = [r for r in good_results if r['correct_lt50'] == r['total_lt50']]
    if perfect:
        best = min(perfect, key=lambda x: x['total_loca_avg'])
        print(f"  187/187 중 LOCA 반응시간 최소: {best['param']}")
        print(f"    HL={best['hl_avg']:.1f}s(max {best['hl_max']}s), CL={best['cl_avg']:.1f}s(max {best['cl_max']}s), RCP={best['rcp_avg']:.1f}s(max {best['rcp_max']}s)")

    # 186/187 (t133만 틀리는) 중 최적
    t133_only = [r for r in good_results if r['correct_lt50'] == 186 and 't133' in r['wrong_str']]
    if t133_only:
        best186 = min(t133_only, key=lambda x: x['total_loca_avg'])
        print(f"\n  186/187 (t133만) 중 LOCA 반응시간 최소: {best186['param']}")
        print(f"    HL={best186['hl_avg']:.1f}s(max {best186['hl_max']}s), CL={best186['cl_avg']:.1f}s(max {best186['cl_max']}s), RCP={best186['rcp_avg']:.1f}s(max {best186['rcp_max']}s)")

    print("=" * 140)
