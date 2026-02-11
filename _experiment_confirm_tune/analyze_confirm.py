"""
Confirm count / GRACE_PERIOD 최적 조합 탐색
Team N Code 건드리지 않고 독립 실행.
"""
import sys, os, csv
from pathlib import Path
from collections import defaultdict, Counter
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

# Team N code의 모듈만 import (수정 안 함)
sys.path.insert(0, str(Path('/Users/jangjaewon/Desktop/NAS/Team N code/py')))
from main import LABELS

import tensorflow as tf
import joblib

M_DIR = Path('/Users/jangjaewon/Desktop/NAS/models_9class_v3_ss5')
PREFIX = 'tcn__feat=physics_v3__val=1__ep=100__cw=1__seed=0__skipd=5__dan=1__ss=5__win=3__stride=1'

model = tf.keras.models.load_model(M_DIR / f'{PREFIX}__model.keras', compile=False)
scaler = joblib.load(M_DIR / f'{PREFIX}__scaler.pkl')
feat = joblib.load(M_DIR / f'{PREFIX}__feature_transformer.pkl')
raw_names = feat.feature_names_all

from main import RealtimeInference
inf = RealtimeInference(model, scaler, feat, raw_names)
dummy = np.zeros((1, inf.WINDOW, len(feat.feature_names)), dtype=np.float32)
_ = model.predict(dummy, verbose=0)

# ===== 정답 로드 =====
answers = {}
with open('/Users/jangjaewon/Desktop/NAS/data/real_test_data/answers.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = int(row['test_id'])
        answers[tid] = {'label': row['label'], 'delay': int(row['malf_delay'])}

# ===== 200건 raw prediction 수집 =====
data_sources = [
    ('/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100', range(1, 101)),
    ('/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200', range(101, 201)),
]

print('=== 200건 raw prediction history 수집중... ===')
all_histories = []

for data_dir, tid_range in data_sources:
    data_path = Path(data_dir)
    for tid in tid_range:
        if tid not in answers:
            continue
        ans = answers[tid]
        inf.reset()
        header = None

        sec_files = sorted(
            [p for p in data_path.glob(f'test{tid}_sec*.csv') if ' ' not in p.name],
            key=lambda p: int(p.stem.split('sec')[1])
        )
        if not sec_files:
            continue

        for sf in sec_files:
            sec = int(sf.stem.split('sec')[1])
            with open(sf, 'r', encoding='utf-8', newline='') as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                continue
            col_names = rows[0]
            data_row = rows[1]
            if header is None:
                header = col_names
            values = []
            for v in data_row:
                try:
                    val = float(v)
                    if not np.isfinite(val):
                        val = 0.0
                except:
                    val = 0.0
                values.append(val)
            x = np.array(values, dtype=np.float32)
            pred, prob, _ = inf.process_sec(x, sec, header)

        all_histories.append({
            'tid': tid,
            'label': ans['label'],
            'delay': ans['delay'],
            'preds': list(inf.pred_history),
            'probs': [p.copy() for p in inf.prob_history],
        })

print(f'수집 완료: {len(all_histories)}건\n')

# ===== 시뮬레이션 함수 =====
LOCA_INDICES = {1, 2, 3}
LATE_WINDOW = 10
LATE_CONF = 0.6


def simulate(h, loca_confirm, non_loca_confirm, grace):
    preds = h['preds']
    probs = h['probs']
    label = h['label']

    confirmed = False
    confirmed_label = None
    confirmed_idx = None
    confirmed_sec = None
    late_corrected = False

    for i, pred in enumerate(preds):
        sec = i + 1

        if not confirmed and pred != 0 and sec > grace:
            n_confirm = loca_confirm if pred in LOCA_INDICES else non_loca_confirm
            if i + 1 >= n_confirm:
                recent = preds[i - n_confirm + 1: i + 1]
                if len(recent) == n_confirm and all(p == pred for p in recent):
                    confirmed = True
                    confirmed_label = LABELS[pred]
                    confirmed_idx = pred
                    confirmed_sec = sec

        if confirmed and not late_corrected and confirmed_idx in LOCA_INDICES:
            if i + 1 >= LATE_WINDOW:
                recent_preds = preds[i - LATE_WINDOW + 1: i + 1]
                recent_probs = probs[i - LATE_WINDOW + 1: i + 1]
                loca_preds = [p for p in recent_preds if p in LOCA_INDICES]
                if loca_preds:
                    counter = Counter(loca_preds)
                    most_idx, most_count = counter.most_common(1)[0]
                    if most_idx != confirmed_idx:
                        ratio = most_count / len(loca_preds)
                        avg_conf = float(np.mean([p[most_idx] for p in recent_probs]))
                        if ratio >= 0.5 and avg_conf >= LATE_CONF:
                            late_corrected = True
                            confirmed_label = LABELS[most_idx]
                            confirmed_idx = most_idx

    final = confirmed_label or 'NORMAL'
    ok = (final == label)
    return ok, confirmed_sec, final


# ===== 1단계: GRACE도 포함한 전수 탐색 =====
print('=' * 100)
print('  GRACE / LOCA_confirm / 기타_confirm 전수 탐색')
print('=' * 100)

best_score = 0
best_combos = []

for grace in range(2, 7):
    for loca_c in range(2, 7):
        for non_loca_c in range(2, 5):
            correct = 0
            react_times = []
            for h in all_histories:
                ok, csec, _ = simulate(h, loca_c, non_loca_c, grace)
                if ok:
                    correct += 1
                if csec and h['delay'] > 0:
                    react_times.append(csec - h['delay'])

            avg_react = np.mean(react_times) if react_times else 99

            if correct > best_score:
                best_score = correct
                best_combos = [(grace, loca_c, non_loca_c, correct, avg_react)]
            elif correct == best_score:
                best_combos.append((grace, loca_c, non_loca_c, correct, avg_react))

print(f'\n  최고 정확도: {best_score}/200')
print(f'\n  동일 정확도 조합 ({len(best_combos)}개):')
print(f'  {"GRACE":>6s} {"LOCA":>6s} {"기타":>6s} | {"정확도":>10s} | {"평균반응":>8s}')
print(f'  {"-"*55}')

best_combos.sort(key=lambda x: x[4])  # 반응시간 빠른 순
for g, lc, nc, corr, ar in best_combos:
    marker = ' ◀ 현재' if g == 5 and lc == 5 and nc == 3 else ''
    print(f'  {g:>6d} {lc:>6d} {nc:>6d} | {corr:>3d}/200 {100*corr/200:.1f}% | {ar:>6.1f}s{marker}')


# ===== 2단계: LOCA=2~3에서 추가 오답 분석 =====
print('\n' + '=' * 100)
print('  LOCA=2~3에서 추가로 틀리는 케이스 분석')
print('=' * 100)

# 현재 설정 (GRACE=5, LOCA=5, 기타=3)
current_results = {}
for h in all_histories:
    ok, csec, final = simulate(h, 5, 3, 5)
    current_results[h['tid']] = (ok, final)

for loca_c in [2, 3]:
    for grace in [3, 4, 5]:
        new_errors = []
        fixed = []
        for h in all_histories:
            ok_new, csec_new, final_new = simulate(h, loca_c, 2, grace)
            ok_cur = current_results[h['tid']][0]

            if ok_cur and not ok_new:
                new_errors.append((h['tid'], h['label'], h['delay'], final_new))
            if not ok_cur and ok_new:
                fixed.append((h['tid'], h['label'], h['delay'], final_new))

        if new_errors or fixed:
            total_ok = sum(1 for h in all_histories
                          for ok, _, _ in [simulate(h, loca_c, 2, grace)] if ok)
            print(f'\n  [GRACE={grace}, LOCA={loca_c}, 기타=2] → {total_ok}/200')
            if new_errors:
                print(f'    새로 틀림 ({len(new_errors)}건):')
                for tid, label, delay, pred in new_errors:
                    print(f'      test{tid}: 정답={label}, 예측={pred}, delay={delay}')
            if fixed:
                print(f'    새로 맞음 ({len(fixed)}건):')
                for tid, label, delay, pred in fixed:
                    print(f'      test{tid}: 정답={label}, 예측={pred}, delay={delay}')


# ===== 3단계: 최적 후보 vs 현재 상세 비교 =====
print('\n' + '=' * 100)
print('  최적 후보 vs 현재 설정 상세 비교')
print('=' * 100)

fastest = best_combos[0]  # 반응시간 가장 빠른 최고 정확도
g, lc, nc = fastest[0], fastest[1], fastest[2]

print(f'\n  최적 후보: GRACE={g}, LOCA={lc}, 기타={nc}')
print(f'  현재 설정: GRACE=5, LOCA=5, 기타=3')

print(f'\n  {"사고유형":>14s} | {"현재반응":>8s} | {"최적반응":>8s} | {"차이":>6s}')
print(f'  {"-"*50}')

for label in LABELS:
    if label == 'NORMAL':
        continue

    cur_reacts = []
    new_reacts = []
    for h in all_histories:
        if h['label'] != label:
            continue
        _, csec_cur, _ = simulate(h, 5, 3, 5)
        _, csec_new, _ = simulate(h, lc, nc, g)
        delay = h['delay']
        if csec_cur and delay > 0:
            cur_reacts.append(csec_cur - delay)
        if csec_new and delay > 0:
            new_reacts.append(csec_new - delay)

    cur_avg = np.mean(cur_reacts) if cur_reacts else 0
    new_avg = np.mean(new_reacts) if new_reacts else 0
    diff = new_avg - cur_avg

    print(f'  {label:>14s} | {cur_avg:>6.1f}s | {new_avg:>6.1f}s | {diff:>+5.1f}s')
