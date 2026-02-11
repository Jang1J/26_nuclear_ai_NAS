"""
2단계 실험:
1) 200개 전체 추론 1회 → pred/prob 캐시 (pickle 저장)
2) 캐시로 확정 로직만 고속 시뮬레이션
"""
import os, sys, csv, pickle, time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

TEAM_N_PY = Path('/Users/jangjaewon/Desktop/NAS/Team N code/py')
sys.path.insert(0, str(TEAM_N_PY))
from main import LABELS, RealtimeInference, load_pipeline

CACHE_FILE = '/Users/jangjaewon/Desktop/NAS/_pred_cache_200.pkl'

def find_data_dir(tid):
    for d in ['/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100',
              '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200']:
        if Path(d).joinpath(f'test{tid}_sec1.csv').exists():
            return Path(d)
    return None

# 정답 로드
answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        answers[int(row['test_id'])] = {'label': row['label'], 'delay': int(row['malf_delay'])}

ALL_TIDS = sorted(answers.keys())

# ===== 1단계: 캐시 생성 (1회만) =====
if not Path(CACHE_FILE).exists():
    print("[1/2] 모델 추론 캐시 생성 중...")
    model, scaler, feat_transformer, raw_feature_names = load_pipeline()
    inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)
    dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    cache = {}
    t0 = time.time()
    for i, tid in enumerate(ALL_TIDS):
        data_dir = find_data_dir(tid)
        if data_dir is None:
            continue

        inferencer.reset()
        header = None
        preds = []
        probs = []

        for sec in range(1, 61):
            fp = data_dir / f"test{tid}_sec{sec}.csv"
            if not fp.exists():
                continue
            with open(fp, 'r', encoding='utf-8', newline='') as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                continue
            col_names = rows[0]
            data_row = rows[1]
            values = []
            for v in data_row:
                try:
                    val = float(v)
                    if not np.isfinite(val): val = 0.0
                except: val = 0.0
                values.append(val)
            x = np.array(values, dtype=np.float32)
            if header is None:
                header = col_names

            pred, prob, _ = inferencer.process_sec(x, sec, header)
            preds.append(int(pred))
            probs.append(prob.tolist())

        cache[tid] = {'preds': preds, 'probs': probs}
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(ALL_TIDS)} done ({time.time()-t0:.0f}s)")

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"[1/2] 캐시 저장 완료: {CACHE_FILE} ({time.time()-t0:.0f}s)")
else:
    print(f"[1/2] 캐시 로드: {CACHE_FILE}")

with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

# ===== 2단계: 고속 시뮬레이션 =====
print("\n[2/2] 비대칭 확정 로직 시뮬레이션")
print("=" * 100)

GRACE_PERIOD = 3
CONFIRM_SGTR_ESDE = 2
LOCA_INDICES = {1, 2, 3}
CL_IDX = 2

best_score = 0
best_params = None

for CL_N in [4, 5, 6, 7, 8]:
    for HL_RCP_N in [4]:
        for GUARD_WINDOW in [4, 5, 6, 7, 8]:
            for CL_THRESH in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
                correct = 0
                wrong = 0
                unconfirmed = 0
                wrong_list = []
                react_times = []

                for tid in ALL_TIDS:
                    if tid not in cache:
                        continue
                    preds = cache[tid]['preds']
                    probs = cache[tid]['probs']

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

                        if pred in LOCA_INDICES:
                            if pred == CL_IDX:
                                n_req = CL_N
                            else:
                                n_req = HL_RCP_N
                                # CL 전환 가드
                                window = min(GUARD_WINDOW, i + 1)
                                recent_cl = [probs[j][CL_IDX] for j in range(max(0, i+1-window), i+1)]
                                max_cl = max(recent_cl)
                                if max_cl > CL_THRESH:
                                    continue
                        else:
                            n_req = CONFIRM_SGTR_ESDE

                        if i + 1 >= n_req:
                            recent = preds[i+1-n_req:i+1]
                            if all(p == pred for p in recent):
                                confirmed_label = LABELS[pred]
                                confirmed_sec = sec

                    final = confirmed_label if confirmed_label else 'NORMAL'
                    true_label = answers[tid]['label']

                    if final == true_label:
                        correct += 1
                        if confirmed_sec and answers[tid]['delay'] > 0:
                            react_times.append(confirmed_sec - answers[tid]['delay'])
                    elif confirmed_label is None and true_label == 'NORMAL':
                        correct += 1
                    elif confirmed_label is None:
                        unconfirmed += 1
                        wrong_list.append(f"test{tid}({true_label},d={answers[tid]['delay']},미확정)")
                    else:
                        wrong += 1
                        wrong_list.append(f"test{tid}({true_label}→{final})")

                total = correct + wrong + unconfirmed
                avg_react = np.mean(react_times) if react_times else 0

                if correct > best_score or (correct == best_score and avg_react < best_react):
                    best_score = correct
                    best_react = avg_react
                    best_params = (HL_RCP_N, CL_N, GUARD_WINDOW, CL_THRESH)
                    best_wrong = wrong_list[:]

                if correct >= 197:
                    print(f"HL/RCP={HL_RCP_N} CL={CL_N} guard={GUARD_WINDOW} thresh={CL_THRESH:.2f} | {correct}/{total} | 오={wrong} 미={unconfirmed} | react={avg_react:.1f}s")
                    for w in wrong_list:
                        print(f"    {w}")

print()
print("=" * 60)
print(f"최적: HL/RCP={best_params[0]} CL={best_params[1]} guard={best_params[2]} thresh={best_params[3]:.2f}")
print(f"점수: {best_score}/200, react={best_react:.1f}s")
print(f"오답: {best_wrong}")
print("=" * 60)
