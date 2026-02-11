"""test133 초별 상세 분석"""
import csv, pickle
import numpy as np

CACHE_FILE = '/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl'

with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

LABELS = ['NORMAL', 'LOCA_HL', 'LOCA_CL', 'LOCA_RCP',
          'SGTR_Loop1', 'SGTR_Loop2', 'SGTR_Loop3',
          'ESDE_in', 'ESDE_out']

preds = cache[133]['preds']
probs = cache[133]['probs']

print("test133 | 정답: LOCA_CL | delay=27")
print("=" * 120)
print(f"{'sec':>4s} | {'pred':>10s} | {'NORMAL':>8s} | {'HL':>8s} | {'CL':>8s} | {'RCP':>8s} | {'SGTR1':>8s} | {'SGTR2':>8s} | {'SGTR3':>8s} | {'ESDEi':>8s} | {'ESDEo':>8s} | 비고")
print("-" * 120)

streak = []
for i in range(len(preds)):
    sec = i + 1
    pred = preds[i]
    prob = probs[i]

    label = LABELS[pred]

    if pred != 0:
        if len(streak) > 0 and streak[-1][0] == pred:
            streak.append((pred, sec))
        else:
            streak = [(pred, sec)]
    else:
        streak = []

    # 비고
    note = ""
    if sec == 27:
        note = "<-- delay (사고시작)"
    if pred != 0:
        same_count = len(streak)
        note += f" [{LABELS[pred][-2:]}x{same_count}]"

    # 사고 전후만 출력 (sec 25~60)
    if sec >= 25:
        prob_str = " | ".join(f"{p:.6f}" for p in prob)
        print(f"{sec:>4d} | {label:>10s} | {prob_str} | {note}")
