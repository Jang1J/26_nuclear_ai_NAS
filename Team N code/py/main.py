"""
대회 제출용 실시간 추론 파이프라인 — Team 6
모델: TCN V3 (100ep, physics_v3 features)

[보강사항]
- CPU-only TF 호환 (GPU 없는 노트북 대응)
- NaN/inf 방어 (데이터 결측 대응)
- 에러 핸들링 (파일 읽기, 모델 추론 안전장치)
- 경로 이식성 (상대 경로만 사용)
"""
import os
import time
import csv
import socket
import sys
import warnings
from pathlib import Path

# ===== CPU-only 모드 (GPU 없는 노트북 대응) =====
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TF 경고 억제

import numpy as np
import joblib

# TF import를 try-except로 감싸서 에러 메시지 개선
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except ImportError as e:
    print(f"[FATAL] TensorFlow import 실패: {e}")
    print("  → pip install tensorflow 실행 필요")
    sys.exit(1)

# ===== 경로 설정 (상대 경로 — 이식성 확보) =====
PY_DIR = Path(__file__).resolve().parent        # py/
BASE_DIR = PY_DIR.parent                         # Team N code/
MODEL_DIR = BASE_DIR / "models"

# practice 모듈 import (feature_transformer.pkl unpickle에 필요)
# py/practice/ 디렉토리에 feature_method.py, feature_method_v3.py 있어야 함
sys.path.insert(0, str(PY_DIR))

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]

# ===== 모델 파일 prefix =====
MODEL_PREFIX = "tcn__feat=physics_v3__val=1__ep=100__cw=1__seed=0__skipd=5__dan=1__ss=5__win=3__stride=1"


def load_pipeline():
    """모델, 스케일러, 피처 트랜스포머를 1회 로드."""
    model_path = MODEL_DIR / f"{MODEL_PREFIX}__model.keras"
    scaler_path = MODEL_DIR / f"{MODEL_PREFIX}__scaler.pkl"
    feat_path = MODEL_DIR / f"{MODEL_PREFIX}__feature_transformer.pkl"

    # 파일 존재 확인
    for p in [model_path, scaler_path, feat_path]:
        if not p.exists():
            print(f"[FATAL] 파일 없음: {p}")
            sys.exit(1)

    # 모델 로드
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # 스케일러 로드
    scaler = joblib.load(scaler_path)

    # 피처 트랜스포머 로드
    # ⚠ practice.feature_method_v3.FeaturePhysicsV3 클래스를
    #    unpickle하므로 py/practice/ 모듈이 sys.path에 있어야 함
    feat_transformer = joblib.load(feat_path)

    # 원본 컬럼 이름 (학습 시 사용한 204개)
    raw_feature_names = feat_transformer.feature_names_all

    print(f"[INIT] Model loaded: {MODEL_PREFIX}")
    print(f"[INIT] Raw features: {len(raw_feature_names)}, "
          f"Output features: {len(feat_transformer.feature_names)}")
    print(f"[INIT] TF device: CPU (CUDA_VISIBLE_DEVICES=-1)")

    return model, scaler, feat_transformer, raw_feature_names


def udp_send(sock, targets, test_id, sec, payload):
    """UDP 전송 (실패 시에도 프로세스 중단 없음)."""
    msg = f"test{test_id} sec{sec}|{payload}"
    data = msg.encode("utf-8")
    for ip, port in targets:
        try:
            sock.sendto(data, (ip, port))
        except Exception as e:
            print(f"[UDP_ERR] {ip}:{port} → {e}")
    print("[UDP_SEND]", msg)


def sanitize_array(arr):
    """NaN/inf를 0으로 치환."""
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


# ===== 실시간 추론 클래스 =====
class RealtimeInference:
    """
    매 초 데이터를 받아 버퍼링 → 윈도우 구성 → 추론.
    - 피처 변환은 누적 데이터 전체에 대해 수행 (미분 계산을 위해)
    - 윈도우 크기: 3 (학습과 동일)
    - 진단 확정: 연속 3초 동일 비정상 예측 시 확정
    """

    WINDOW = 3
    CONFIRM_COUNT = 2      # 연속 N초 같은 비정상이면 확정 (SGTR/ESDE)
    LOCA_CONFIRM_COUNT = 4 # LOCA 계열은 더 보수적 확정 (초기 HL↔CL 혼동 방지)
    GRACE_PERIOD = 3       # 처음 N초는 확정 안 함 (패딩 불안정 방지)

    # LOCA 계열 인덱스 (Late Correction 대상)
    LOCA_INDICES = {1, 2, 3}  # LOCA_HL, LOCA_CL, LOCA_RCP
    LATE_WINDOW = 10           # 후기 교정: 마지막 N초 확인
    LATE_CONF = 0.6            # 후기 교정: 최소 평균 confidence

    def __init__(self, model, scaler, feat_transformer, raw_feature_names):
        self.model = model
        self.scaler = scaler
        self.feat = feat_transformer
        self.raw_feature_names = raw_feature_names

        # col_to_idx 캐시 (헤더가 바뀌지 않으면 재사용)
        self._cached_header = None
        self._cached_col_to_idx = None

        # 상태 초기화
        self.reset()

    def reset(self):
        """새 테스트 시작 시 초기화."""
        self.buffer = []
        self.pred_history = []
        self.prob_history = []  # Late Correction용 확률 히스토리
        self.confirmed = False
        self.confirmed_label = None
        self.confirmed_idx = None
        self.confirmed_time = None
        self.late_corrected = False  # 교정 1회만 허용

    def _get_col_mapping(self, col_names):
        """컬럼 인덱스 매핑 (캐시)."""
        if self._cached_header is not col_names and self._cached_header != col_names:
            self._cached_col_to_idx = {name: i for i, name in enumerate(col_names)}
            self._cached_header = col_names
            # 매핑 확인
            found = sum(1 for n in self.raw_feature_names
                        if n in self._cached_col_to_idx)
            print(f"  [COL_MAP] {found}/{len(self.raw_feature_names)} features matched "
                  f"(from {len(col_names)} competition columns)")
        return self._cached_col_to_idx

    def process_sec(self, x_raw, sec, col_names):
        """
        1초 데이터를 받아 추론.

        Args:
            x_raw: np.array — 대회 서버에서 온 전체 컬럼 데이터
            sec: int — 현재 시간(초)
            col_names: list — 대회 데이터 컬럼명

        Returns:
            pred: int — 예측 클래스 인덱스
            prob: np.array (9,) — 클래스별 확률
            is_confirmed: bool — 이번 초에 진단 확정됐는지
        """
        try:
            # 0) NaN/inf 방어 — 입력 데이터
            x_raw = sanitize_array(x_raw.astype(np.float32))

            # 1) 대회 컬럼 → 학습용 204컬럼 추출
            col_to_idx = self._get_col_mapping(col_names)
            x_selected = np.array(
                [x_raw[col_to_idx[name]] if name in col_to_idx else 0.0
                 for name in self.raw_feature_names],
                dtype=np.float32
            )
            x_selected = sanitize_array(x_selected)
            self.buffer.append(x_selected)

            # 2) 누적 버퍼에 대해 피처 변환 (미분 계산을 위해 전체 필요)
            X_buf = np.array(self.buffer, dtype=np.float32)  # (t, 204)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_feat, _, _ = self.feat.transform(X_buf, np.zeros(len(X_buf)))

            # NaN/inf 방어 — 피처 변환 후
            X_feat = sanitize_array(X_feat)

            # 3) 스케일링 + 클리핑 (반올림 오차로 인한 폭발 방지)
            X_scaled = self.scaler.transform(X_feat)  # (t, 282)
            X_scaled = sanitize_array(X_scaled)
            X_scaled = np.clip(X_scaled, -10, 10)  # 학습 범위 내로 제한

            # 4) 윈도우 구성 (마지막 WINDOW개)
            t = X_scaled.shape[0]
            if t < self.WINDOW:
                # 윈도우 부족 → 첫 행으로 패딩
                pad = np.tile(X_scaled[0:1], (self.WINDOW - t, 1))
                window = np.vstack([pad, X_scaled])  # (WINDOW, 282)
            else:
                window = X_scaled[t - self.WINDOW: t]  # (WINDOW, 282)

            # 5) 추론
            window_input = window[np.newaxis, ...]  # (1, WINDOW, 282)
            prob = self.model.predict(window_input, verbose=0)[0]  # (9,)

            # NaN/inf 방어 — 모델 출력
            if not np.all(np.isfinite(prob)):
                print(f"  [WARN] sec{sec}: 모델 출력에 NaN 감지 → uniform 분포 대체")
                prob = np.ones(len(LABELS)) / len(LABELS)

            pred = int(np.argmax(prob))

        except Exception as e:
            # 추론 실패 시 안전한 기본값 반환 (NORMAL)
            print(f"  [ERROR] sec{sec}: 추론 오류 → {e}")
            prob = np.zeros(len(LABELS))
            prob[0] = 1.0  # NORMAL
            pred = 0
            self.pred_history.append(pred)
            return pred, prob, False

        # 6) 진단 확정 로직 (연속 N초 동일 비정상 예측)
        self.pred_history.append(pred)
        self.prob_history.append(prob.copy())
        is_confirmed = False

        # GRACE_PERIOD 이내는 확정 안 함 (패딩으로 인한 오탐 방지)
        if (not self.confirmed and pred != 0
                and len(self.pred_history) > self.GRACE_PERIOD):
            # LOCA 계열은 더 보수적 확정 (초기 HL↔CL 혼동 방지)
            n_confirm = self.LOCA_CONFIRM_COUNT if pred in self.LOCA_INDICES else self.CONFIRM_COUNT
            if len(self.pred_history) >= n_confirm:
                recent = self.pred_history[-n_confirm:]
                if all(p == pred for p in recent):
                    self.confirmed = True
                    self.confirmed_label = LABELS[pred]
                    self.confirmed_idx = pred
                    is_confirmed = True

        return pred, prob, is_confirmed

    def get_late_corrected_label(self):
        """
        LOCA HL↔CL 후기 교정.
        확정이 LOCA 계열이고, 최근 예측이 다른 LOCA 하위 클래스로 수렴했으면 교정.
        교정은 1회만 허용.
        """
        if not self.confirmed or self.late_corrected:
            return self.confirmed_label

        if self.confirmed_idx not in self.LOCA_INDICES:
            return self.confirmed_label

        if len(self.pred_history) < self.LATE_WINDOW:
            return self.confirmed_label

        recent_preds = self.pred_history[-self.LATE_WINDOW:]
        recent_probs = self.prob_history[-self.LATE_WINDOW:]

        # LOCA 계열 예측만 추출
        from collections import Counter
        loca_preds = [p for p in recent_preds if p in self.LOCA_INDICES]
        if not loca_preds:
            return self.confirmed_label

        counter = Counter(loca_preds)
        most_idx, most_count = counter.most_common(1)[0]

        if most_idx == self.confirmed_idx:
            return self.confirmed_label  # 교정 불필요

        # 교정 조건: 과반 + 평균 confidence 충족
        ratio = most_count / len(loca_preds)
        avg_conf = float(np.mean([p[most_idx] for p in recent_probs]))

        if ratio >= 0.5 and avg_conf >= self.LATE_CONF:
            corrected = LABELS[most_idx]
            self.late_corrected = True
            self.confirmed_label = corrected
            self.confirmed_idx = most_idx
            print(f"  [LATE_CORRECTION] {LABELS[self.confirmed_idx]} "
                  f"(ratio={ratio:.2f}, avg_conf={avg_conf:.4f})")
            return corrected

        return self.confirmed_label


# ===== 메인 실행 =====
if __name__ == "__main__":
    print("=" * 60)
    print("  Team 6 — 실시간 추론 파이프라인")
    print("  Model: TCN V3 (100ep, physics_v3, 282 features)")
    print("=" * 60)

    # 설정
    DATA_ROOT = BASE_DIR / "data"
    TEST_START, TEST_END = 1, 10
    SEC_START, SEC_END = 1, 60
    POLL_INTERVAL = 0.05

    TARGETS = [("192.168.0.3", 7001)]

    # 모델 로딩 (1회)
    model, scaler, feat_transformer, raw_feature_names = load_pipeline()
    inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)

    # 워밍업 — 첫 추론이 느리므로 dummy로 1회 실행
    print("[INIT] Warmup inference...")
    dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)),
                     dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    print("[INIT] Warmup done.\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for test_id in range(TEST_START, TEST_END + 1):
        folder_path = DATA_ROOT / f"test{test_id}"

        # test{n} 폴더가 생성될 때까지 대기
        while not folder_path.exists():
            time.sleep(POLL_INTERVAL)

        print(f"\n=== TEST {test_id} START ===")

        # 새 테스트마다 초기화
        inferencer.reset()
        diagnostic_results = None
        diagnostic_time = None
        header = None

        for sec in range(SEC_START, SEC_END + 1):
            file_path = folder_path / f"test{test_id}_sec{sec}.csv"

            # sec 파일이 생성될 때까지 대기
            while not file_path.exists():
                time.sleep(POLL_INTERVAL)

            # 파일이 '완성'될 때까지 기다렸다가 읽기
            retry_count = 0
            while True:
                try:
                    if not file_path.exists():
                        time.sleep(POLL_INTERVAL)
                        continue

                    with open(file_path, "r", encoding="utf-8", newline="") as f:
                        rows = list(csv.reader(f))

                    if len(rows) < 2:
                        time.sleep(POLL_INTERVAL)
                        continue

                    col_names = rows[0]
                    data_row = rows[1]

                    if len(data_row) != len(col_names):
                        time.sleep(POLL_INTERVAL)
                        continue

                    if any(v.strip() == "" for v in data_row):
                        time.sleep(POLL_INTERVAL)
                        continue

                    break

                except (IOError, OSError) as e:
                    retry_count += 1
                    if retry_count > 100:
                        print(f"  [WARN] sec{sec}: 파일 읽기 실패 100회 → skip")
                        break
                    time.sleep(POLL_INTERVAL)
            else:
                pass  # 정상 탈출

            if retry_count > 100:
                # 읽기 실패 시 기본값으로 UDP 전송
                probs_str = ",".join(["0.111111"] * 9)
                dr = diagnostic_results if diagnostic_results is not None else "None"
                dt = diagnostic_time if diagnostic_time is not None else 0.0
                payload = f"{dr},{dt},{probs_str}"
                udp_send(sock, TARGETS, test_id, sec, payload)
                continue

            # 파이프라인 시간 측정
            start_time = time.time()

            # 안전한 float 변환 (NaN → 0.0)
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

            # ===== 전처리 + 모델 추론 + 진단 확정 =====
            if header is None:
                header = col_names

            pred, prob, is_confirmed = inferencer.process_sec(x, sec, header)

            runtime = time.time() - start_time

            # 진단 확정 시 1회 수행
            if diagnostic_results is None and is_confirmed:
                diagnostic_results = LABELS[pred]
                diagnostic_time = round(sec + runtime, 1)
                print(f"  *** 진단 확정: {diagnostic_results} at {diagnostic_time}s ***")

            # Late Correction: LOCA HL↔CL 후기 교정 (매 sec 확인)
            if diagnostic_results is not None:
                corrected = inferencer.get_late_corrected_label()
                if corrected != diagnostic_results:
                    print(f"  *** 후기 교정: {diagnostic_results} -> {corrected} at sec{sec} ***")
                    diagnostic_results = corrected

            # UDP 전송
            probs_str = ",".join(f"{float(p):.6f}" for p in prob)
            dr = diagnostic_results if diagnostic_results is not None else "None"
            dt = diagnostic_time if diagnostic_time is not None else 0.0
            payload = f"{dr},{dt},{probs_str}"

            udp_send(sock, TARGETS, test_id, sec, payload)

    sock.close()
    print("\n[DONE]")
