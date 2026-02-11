"""
Team 6 code 최종본 — 실제 모델로 추론, 대회 형식 UDP_SEND 출력.
Team 6 code/py/main.py의 RealtimeInference + udp_send 형식 그대로 재현.
"""
import os, sys, csv, time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Team 6 code에서 직접 import
TEAM6_PY = Path('/Users/jangjaewon/Desktop/NAS/Team 6 code/py')
sys.path.insert(0, str(TEAM6_PY))
from main import RealtimeInference, LABELS, load_pipeline

# ===== 모델 로드 =====
print("모델 로딩 중...")
model, scaler, feat_transformer, raw_feature_names = load_pipeline()
inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)

# 워밍업
dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)), dtype=np.float32)
_ = model.predict(dummy, verbose=0)
print("워밍업 완료\n")


def find_data(tid, dataset):
    """데이터 파일 경로 찾기."""
    if dataset == '200':
        for d in ['/Users/jangjaewon/Desktop/NAS/_test_data/test_1_100',
                  '/Users/jangjaewon/Desktop/NAS/_test_data/test_101-200']:
            if Path(d).joinpath(f'test{tid}_sec1.csv').exists():
                return Path(d)
    elif dataset == '300':
        d = '/Users/jangjaewon/Desktop/NAS/_test_data/test201_300'
        if Path(d).joinpath(f'test{tid}_sec1.csv').exists():
            return Path(d)
    elif dataset == 'cl150':
        d = '/Users/jangjaewon/Desktop/NAS/_test_data/LOCA_CL_test'
        if Path(d).joinpath(f'test{tid}_sec1.csv').exists():
            return Path(d)
    elif dataset == 'dt5':
        d = '/Users/jangjaewon/Desktop/NAS/_test_data/test_dt5_'
        if Path(d).joinpath(f'test{tid}_sec1.csv').exists():
            return Path(d)
    return None


def run_real_demo(tid, dataset, true_label, delay=0, extra_info=""):
    """실제 모델로 추론 — 대회 형식 [UDP_SEND] 출력."""
    data_dir = find_data(tid, dataset)
    if data_dir is None:
        print(f"  데이터 없음: {dataset} test{tid}")
        return

    inferencer.reset()
    header = None
    diagnostic_results = None
    diagnostic_time = None

    print(f"\n=== [{dataset}] TEST {tid} START === 정답: {true_label} | delay: {delay} {extra_info}")

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
        if header is None:
            header = col_names

        start_time = time.time()

        values = []
        for v in data_row:
            try:
                val = float(v)
                if not np.isfinite(val):
                    val = 0.0
            except (ValueError, TypeError):
                val = 0.0
            values.append(val)
        x = np.array(values, dtype=np.float32)

        pred, prob, is_confirmed = inferencer.process_sec(x, sec, header)

        # 진단 확정 시 1회 수행 (main.py 397~400줄과 동일)
        if diagnostic_results is None and is_confirmed:
            diagnostic_results = LABELS[pred]
            diagnostic_time = round(time.time() - start_time + sec, 1)
            print(f"  *** 진단 확정: {diagnostic_results} at {diagnostic_time}s ***")

        # UDP 전송 형식 (main.py 403~408줄과 동일)
        probs_str = ",".join(f"{float(p):.6f}" for p in prob)
        dr = diagnostic_results if diagnostic_results is not None else "None"
        dt = diagnostic_time if diagnostic_time is not None else 0.0
        payload = f"{dr},{dt},{probs_str}"

        # [UDP_SEND] 출력 (main.py udp_send 함수와 동일 형식)
        msg = f"test{tid} sec{sec}|{payload}"
        print("[UDP_SEND]", msg)

    # 결과 요약
    final = diagnostic_results if diagnostic_results else "NORMAL"
    ok = "O" if final == true_label else "X"
    print(f"=== TEST {tid} END === 결과: {ok} | 정답={true_label} | 예측={final}")


# ===== 실행: 대표 케이스 =====
print("=" * 60)
print("  Team 6 최종본 — 실제 모델 추론 (대회 형식 출력)")
print("  CL Guard: GW=5, T=0.15 | ESDE Guard: GW=3, T=0.05")
print("=" * 60)

cases = [
    # (tid, dataset, true_label, delay, extra_info)
    (1,   '200',  'SGTR_Loop1', 38, "SGTR 정상 확정"),
    (50,  '200',  'ESDE_in',     6, "ESDE 정상 확정(가드OK)"),
    (57,  '200',  'LOCA_CL',     9, "CL Guard 작동"),
    (88,  'cl150','LOCA_CL',     0, "(leak=1120,n=8) ESDE Guard 작동"),
    (124, 'cl150','LOCA_CL',     0, "(leak=700,n=14) ESDE Guard 작동"),
    (133, '200',  'LOCA_CL',    27, "유일한 대회급 오답"),
]

for tid, dataset, true_label, delay, extra in cases:
    run_real_demo(tid, dataset, true_label, delay, extra)
