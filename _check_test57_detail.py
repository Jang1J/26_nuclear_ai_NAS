"""test57 초별 상세 - 왜 Risk Gating에서도 안 잡히는지"""
import pickle
import numpy as np

CACHE_FILE = '/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl'
with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

LABELS = ['NORMAL', 'LOCA_HL', 'LOCA_CL', 'LOCA_RCP',
          'SGTR_Loop1', 'SGTR_Loop2', 'SGTR_Loop3',
          'ESDE_in', 'ESDE_out']

for tid in [57, 72]:
    preds = cache[tid]['preds']
    probs = cache[tid]['probs']

    print(f"\ntest{tid} | 정답: LOCA_CL | delay={9 if tid==57 else 40}")
    print("=" * 100)
    print(f"{'sec':>4s} | {'pred':>10s} | {'p(HL)':>8s} | {'p(CL)':>8s} | {'p(RCP)':>8s} | {'top1':>8s} | {'margin':>8s} | 비고")
    print("-" * 100)

    streak = []
    for i in range(len(preds)):
        sec = i + 1
        pred = preds[i]
        prob = probs[i]

        sorted_p = sorted(prob, reverse=True)
        top1 = sorted_p[0]
        margin = sorted_p[0] - sorted_p[1]

        if pred != 0:
            if len(streak) > 0 and streak[-1] == pred:
                streak.append(pred)
            else:
                streak = [pred]
        else:
            streak = []

        note = ""
        if pred != 0:
            note = f"[{LABELS[pred][-2:]}x{len(streak)}]"

        delay = 9 if tid == 57 else 40
        if sec >= delay - 1:
            print(f"{sec:>4d} | {LABELS[pred]:>10s} | {prob[1]:.6f} | {prob[2]:.6f} | {prob[3]:.6f} | {top1:.4f} | {margin:.4f} | {note}")
