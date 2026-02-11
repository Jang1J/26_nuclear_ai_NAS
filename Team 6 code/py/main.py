"""
대회 제출용 실시간 추론 파이프라인 — Team 6
모델: TCN V3 (100ep, physics_v3 features)
로직: CL Guard + ESDE Guard (LC 없음)
  - CL Guard: HL/RCP 4연속 확정 시 최근 5초 CL 확률 > 0.15 → 보류
  - ESDE Guard: ESDE 2연속 확정 시 최근 3초 LOCA 확률 > 0.05 → 보류
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import joblib

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
except ImportError as e:
    print(f"[FATAL] TensorFlow import 실패: {e}")
    print("  → pip install tensorflow 실행 필요")
    sys.exit(1)

# ===== 경로 설정 (상대 경로 — 이식성 확보) =====
PY_DIR = Path(__file__).resolve().parent        # py/
BASE_DIR = PY_DIR.parent                        # Team 6 code/
MODEL_DIR = BASE_DIR / "models"

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

    for p in [model_path, scaler_path, feat_path]:
        if not p.exists():
            print(f"[FATAL] 파일 없음: {p}")
            sys.exit(1)

    model = tf.keras.models.load_model(str(model_path), compile=False)
    scaler = joblib.load(scaler_path)
    feat_transformer = joblib.load(feat_path)
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
    진단 확정: CL Guard + ESDE Guard (LC 없음)
    - SGTR: 2연속 확정
    - ESDE: 2연속 확정 + ESDE 가드 (최근 3초 LOCA 확률합 > 0.05 → 보류)
    - LOCA_CL: 4연속 확정
    - LOCA_HL/RCP: 4연속 확정 + CL 가드 (최근 5초 max(CL prob) > 0.15 → 보류)
    """

    WINDOW = 3
    CONFIRM_COUNT = 2      # SGTR/ESDE 연속 확정
    LOCA_CONFIRM_COUNT = 4 # LOCA 계열 연속 확정
    GRACE_PERIOD = 3       # 처음 N초는 확정 안 함

    # LOCA 계열 인덱스
    LOCA_INDICES = {1, 2, 3}  # LOCA_HL, LOCA_CL, LOCA_RCP
    CL_IDX = 2                # LOCA_CL 인덱스
    ESDE_INDICES = {7, 8}     # ESDE_in, ESDE_out

    # CL Guard 파라미터
    CL_GUARD_WINDOW = 5    # 최근 N초의 CL 확률 확인
    CL_GUARD_THRESH = 0.15 # CL 확률 임계값

    # ESDE Guard 파라미터
    ESDE_GUARD_WINDOW = 3     # 최근 N초의 LOCA 확률 확인
    ESDE_GUARD_THRESH = 0.05  # LOCA 확률합 임계값

    def __init__(self, model, scaler, feat_transformer, raw_feature_names):
        self.model = model
        self.scaler = scaler
        self.feat = feat_transformer
        self.raw_feature_names = raw_feature_names

        self._cached_header = None
        self._cached_col_to_idx = None

        self.reset()

    def reset(self):
        """새 테스트 시작 시 초기화."""
        self.buffer = []
        self.pred_history = []
        self.prob_history = []
        self.confirmed = False
        self.confirmed_label = None
        self.confirmed_idx = None

    def _get_col_mapping(self, col_names):
        """컬럼 인덱스 매핑 (캐시)."""
        if self._cached_header is not col_names and self._cached_header != col_names:
            self._cached_col_to_idx = {name: i for i, name in enumerate(col_names)}
            self._cached_header = col_names
            found = sum(1 for n in self.raw_feature_names if n in self._cached_col_to_idx)
            print(f"  [COL_MAP] {found}/{len(self.raw_feature_names)} features matched "
                  f"(from {len(col_names)} competition columns)")
        return self._cached_col_to_idx

    def process_sec(self, x_raw, sec, col_names):
        """
        1초 데이터를 받아 추론.

        Returns:
            pred: int — 예측 클래스 인덱스
            prob: np.array (9,) — 클래스별 확률
            is_confirmed: bool — 이번 초에 진단 확정됐는지
        """
        try:
            x_raw = sanitize_array(x_raw.astype(np.float32))

            col_to_idx = self._get_col_mapping(col_names)
            x_selected = np.array(
                [x_raw[col_to_idx[name]] if name in col_to_idx else 0.0
                 for name in self.raw_feature_names],
                dtype=np.float32
            )
            x_selected = sanitize_array(x_selected)
            self.buffer.append(x_selected)

            X_buf = np.array(self.buffer, dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_feat, _, _ = self.feat.transform(X_buf, np.zeros(len(X_buf)))

            X_feat = sanitize_array(X_feat)

            X_scaled = self.scaler.transform(X_feat)
            X_scaled = sanitize_array(X_scaled)
            X_scaled = np.clip(X_scaled, -10, 10)

            t = X_scaled.shape[0]
            if t < self.WINDOW:
                pad = np.tile(X_scaled[0:1], (self.WINDOW - t, 1))
                window = np.vstack([pad, X_scaled])
            else:
                window = X_scaled[t - self.WINDOW: t]

            window_input = window[np.newaxis, ...]
            prob = self.model.predict(window_input, verbose=0)[0]

            if not np.all(np.isfinite(prob)):
                print(f"  [WARN] sec{sec}: 모델 출력에 NaN 감지 → uniform 분포 대체")
                prob = np.ones(len(LABELS)) / len(LABELS)

            pred = int(np.argmax(prob))

        except Exception as e:
            print(f"  [ERROR] sec{sec}: 추론 오류 → {e}")
            prob = np.zeros(len(LABELS))
            prob[0] = 1.0
            pred = 0
            self.pred_history.append(pred)
            self.prob_history.append(prob.copy())
            return pred, prob, False

        # 6) 진단 확정 로직: CL Guard + ESDE Guard (LC 없음)
        self.pred_history.append(pred)
        self.prob_history.append(prob.copy())
        is_confirmed = False

        if (not self.confirmed and pred != 0
                and len(self.pred_history) > self.GRACE_PERIOD):

            if pred in self.LOCA_INDICES:
                n_confirm = self.LOCA_CONFIRM_COUNT  # 4연속

                if len(self.pred_history) >= n_confirm:
                    recent = self.pred_history[-n_confirm:]
                    if all(p == pred for p in recent):
                        # HL/RCP일 때: CL 가드 체크
                        if pred != self.CL_IDX:
                            w = min(self.CL_GUARD_WINDOW, len(self.prob_history))
                            recent_cl_probs = [self.prob_history[-j-1][self.CL_IDX] for j in range(w)]
                            max_cl_prob = max(recent_cl_probs)

                            if max_cl_prob > self.CL_GUARD_THRESH:
                                pass  # CL 위험 → 확정 보류
                            else:
                                self.confirmed = True
                                self.confirmed_label = LABELS[pred]
                                self.confirmed_idx = pred
                                is_confirmed = True
                        else:
                            # CL 4연속 → 바로 확정
                            self.confirmed = True
                            self.confirmed_label = LABELS[pred]
                            self.confirmed_idx = pred
                            is_confirmed = True

            elif pred in self.ESDE_INDICES:
                # ESDE: 2연속 확정 + LOCA 확률 가드
                n_confirm = self.CONFIRM_COUNT
                if len(self.pred_history) >= n_confirm:
                    recent = self.pred_history[-n_confirm:]
                    if all(p == pred for p in recent):
                        w = min(self.ESDE_GUARD_WINDOW, len(self.prob_history))
                        recent_loca_max = 0
                        for j in range(1, w + 1):
                            loca_sum = sum(self.prob_history[-j][k] for k in self.LOCA_INDICES)
                            recent_loca_max = max(recent_loca_max, loca_sum)

                        if recent_loca_max > self.ESDE_GUARD_THRESH:
                            pass  # LOCA 위험 → 확정 보류
                        else:
                            self.confirmed = True
                            self.confirmed_label = LABELS[pred]
                            self.confirmed_idx = pred
                            is_confirmed = True

            else:
                # SGTR: 2연속 확정
                n_confirm = self.CONFIRM_COUNT
                if len(self.pred_history) >= n_confirm:
                    recent = self.pred_history[-n_confirm:]
                    if all(p == pred for p in recent):
                        self.confirmed = True
                        self.confirmed_label = LABELS[pred]
                        self.confirmed_idx = pred
                        is_confirmed = True

        return pred, prob, is_confirmed


# ===== 메인 실행 =====
if __name__ == "__main__":
    print("=" * 60)
    print("  Team 6 — 실시간 추론 파이프라인")
    print("  Model: TCN V3 (100ep, physics_v3, 282 features)")
    print("  Logic: CL Guard + ESDE Guard (LC 없음)")
    print("=" * 60)

    # ===== 설정 =====
    DATA_ROOT = BASE_DIR / "data"
    TEST_START, TEST_END = 1, 9
    SEC_START, SEC_END = 1, 60
    POLL_INTERVAL = 0.05

    TARGETS = [("192.168.0.3", 7001)]

    # ===== 모델 로딩 (1회) =====
    model, scaler, feat_transformer, raw_feature_names = load_pipeline()
    inferencer = RealtimeInference(model, scaler, feat_transformer, raw_feature_names)

    # ===== 워밍업 =====
    print("[INIT] Warmup inference...")
    dummy = np.zeros((1, inferencer.WINDOW, len(feat_transformer.feature_names)), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    print("[INIT] Warmup done.\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    for test_id in range(TEST_START, TEST_END + 1):
        folder_path = DATA_ROOT / f"test{test_id}"

        # 1) test{n} 폴더가 생성될 때까지 대기
        while not folder_path.exists():
            time.sleep(POLL_INTERVAL)

        # (권장) 이전 잔재 제거: stale sec 파일 오염 방지
        for fp in folder_path.glob(f"test{test_id}_sec*.csv"):
            try:
                fp.unlink()
            except OSError:
                pass

        print(f"\n=== TEST {test_id} START ===")

        # 새 테스트마다 초기화
        inferencer.reset()
        diagnostic_results = None
        diagnostic_time = None
        header = None

        for sec in range(SEC_START, SEC_END + 1):
            file_path = folder_path / f"test{test_id}_sec{sec}.csv"

            # 2) sec 파일이 생성될 때까지 대기
            while not file_path.exists():
                time.sleep(POLL_INTERVAL)

            # 3) "파일을 발견한 순간"부터 파이프라인 시간 측정 시작 (멘토 스켈레톤 기준)
            t_file = time.time()

            # 4) 파일이 '완성'될 때까지 기다렸다가 읽기
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

                except (IOError, OSError):
                    retry_count += 1
                    if retry_count > 100:
                        print(f"  [WARN] sec{sec}: 파일 읽기 실패 100회 → skip")
                        break
                    time.sleep(POLL_INTERVAL)

            # 5) 읽기 실패 시: None + uniform 확률로 전송
            if retry_count > 100:
                probs_str = ",".join(["0.111111"] * 9)
                dr = diagnostic_results if diagnostic_results is not None else "None"
                dt = diagnostic_time if diagnostic_time is not None else 0.0
                payload = f"{dr},{dt},{probs_str}"
                udp_send(sock, TARGETS, test_id, sec, payload)
                continue

            # 6) 안전한 float 변환 (NaN/inf/에러 → 0.0)
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

            # ===== 전처리 + 모델 추론 + 진단 확정 (추론 로직은 그대로) =====
            if header is None:
                header = col_names

            pred, prob, is_confirmed = inferencer.process_sec(x, sec, header)

            # 7) runtime = 파일 발견 시점부터 지금까지 (멘토 기준)
            runtime = time.time() - t_file

            # 8) 진단 확정 시 1회 수행
            if diagnostic_results is None and is_confirmed:
                diagnostic_results = LABELS[pred]
                diagnostic_time = round(sec + runtime, 1)  # 요구사항: sec + runtime (소수 1자리)
                print(f"  *** 진단 확정: {diagnostic_results} at {diagnostic_time}s ***")

            # 9) UDP 전송
            probs_str = ",".join(f"{float(p):.6f}" for p in prob)
            dr = diagnostic_results if diagnostic_results is not None else "None"
            dt = diagnostic_time if diagnostic_time is not None else 0.0
            payload = f"{dr},{dt},{probs_str}"
            udp_send(sock, TARGETS, test_id, sec, payload)

    sock.close()
    print("\n[DONE]")
