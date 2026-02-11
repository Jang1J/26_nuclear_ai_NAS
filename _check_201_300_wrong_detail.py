"""201~300에서 LOCA_CL 오답 케이스 상세 분석 (test235, 250, 255, 271)"""
import pickle, csv
import numpy as np

CACHE_FILE = '/Users/jangjaewon/Desktop/NAS/_pred_cache_201_300.pkl'
with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

answers = {}
with open('/Users/jangjaewon/Desktop/NAS/_test_data/test201_300/answers.csv') as f:
    for row in csv.DictReader(f):
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay']),
                                         'malf_option': row['malf_option']}

LABELS = ['NORMAL', 'LOCA_HL', 'LOCA_CL', 'LOCA_RCP',
          'SGTR_Loop1', 'SGTR_Loop2', 'SGTR_Loop3',
          'ESDE_in', 'ESDE_out']

for tid in [235, 250, 255, 271]:
    if tid not in cache:
        continue
    info = answers[tid]
    preds = cache[tid]['preds']
    probs = cache[tid]['probs']

    print(f"\n{'='*100}")
    print(f"test{tid} | 정답: {info['label']} | delay={info['delay']} | malf_option={info['malf_option']}")
    print(f"{'='*100}")
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

        delay = info['delay']
        if sec >= max(delay - 2, 1):
            print(f"{sec:>4d} | {LABELS[pred]:>10s} | {prob[1]:.6f} | {prob[2]:.6f} | {prob[3]:.6f} | {top1:.4f} | {margin:.4f} | {note}")

# 전체 LOCA_CL 12건 요약
print(f"\n{'='*100}")
print(f"  LOCA_CL 12건 전체 요약")
print(f"{'='*100}")
print(f"{'tid':>5s} | {'delay':>5s} | {'option':>8s} | {'추정leak':>8s} | 초기pred | CL전환sec")
print("-" * 70)

for tid in sorted(answers.keys()):
    if answers[tid]['label'] != 'LOCA_CL':
        continue
    info = answers[tid]
    preds = cache[tid]['preds']
    probs = cache[tid]['probs']
    delay = info['delay']

    # 초기 예측 (사고 직후 첫 비정상)
    first_abnormal = None
    cl_start = None
    for i in range(len(preds)):
        sec = i + 1
        if preds[i] != 0 and sec > delay and first_abnormal is None:
            first_abnormal = LABELS[preds[i]]
        if preds[i] == 2 and sec > delay and cl_start is None:
            cl_start = sec

    # malf_option에서 leak 추정
    opt = info['malf_option']
    # prefix 패턴으로 leak 추정
    if opt.startswith('1') and len(opt) <= 5:
        leak_est = opt[1:]  # 1XXXX → leak=XXXX
    elif opt.startswith('8'):
        leak_est = opt[1:]
    elif opt.startswith('14'):
        leak_est = opt[2:]
    else:
        leak_est = '?'
    leak_est = leak_est.lstrip('0') or '0'

    print(f"  {tid:>3d} | {delay:>5d} | {opt:>8s} | {leak_est:>8s} | {first_abnormal or 'N/A':>8s} | {cl_start or 'Never':>5}")
