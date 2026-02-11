"""
Late Correction 후처리 테스트.

핵심 아이디어:
  초반 확정 후에도 계속 추론 → 마지막 N초의 다수결이 확정과 다르면 교정.
  단, LOCA 계열(HL/CL/RCP) 내에서만 교정 (다른 대분류로는 안 바꿈).

5ep과 100ep 모두 테스트.
"""
import os, sys, csv, warnings
from pathlib import Path
from collections import defaultdict, Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

NAS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NAS_DIR / "Team N code" / "py"))

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]

# LOCA 계열 인덱스
LOCA_INDICES = {1, 2, 3}  # LOCA_HL, LOCA_CL, LOCA_RCP

MODEL_DIR = NAS_DIR / "models_9class_v3_ss5"
DATA_ROOT = NAS_DIR / "Team N code" / "data"
ANSWERS_PATH = NAS_DIR / "data" / "real_test_data" / "answers.csv"


def sanitize_array(arr):
    return np.where(np.isfinite(arr), arr, 0.0)


class InferenceWithLateCorrection:
    """확정 후에도 추론 계속 + 후기 교정."""

    WINDOW = 3
    CONFIRM_COUNT = 3
    GRACE_PERIOD = 5

    def __init__(self, model, scaler, feat, raw_names, late_window=10, late_conf_threshold=0.7):
        self.model = model
        self.scaler = scaler
        self.feat = feat
        self.raw_feature_names = raw_names
        self.late_window = late_window  # 마지막 N초 확인
        self.late_conf_threshold = late_conf_threshold  # 교정 최소 confidence
        self._cached_header = None
        self._cached_col_to_idx = None
        self.reset()

    def reset(self):
        self.buffer = []
        self.pred_history = []      # 모든 sec의 예측
        self.prob_history = []      # 모든 sec의 확률
        self.confirmed = False
        self.confirmed_label = None
        self.confirmed_idx = None   # 확정된 클래스 인덱스
        self.confirm_sec = None

    def _get_col_mapping(self, col_names):
        if self._cached_header != col_names:
            self._cached_col_to_idx = {name: i for i, name in enumerate(col_names)}
            self._cached_header = col_names
        return self._cached_col_to_idx

    def process_sec(self, x_raw, sec, col_names):
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

            prob = self.model.predict(window[np.newaxis, ...], verbose=0)[0]
            if not np.all(np.isfinite(prob)):
                prob = np.ones(len(LABELS)) / len(LABELS)
            pred = int(np.argmax(prob))
        except Exception as e:
            prob = np.zeros(len(LABELS)); prob[0] = 1.0; pred = 0

        self.pred_history.append(pred)
        self.prob_history.append(prob.copy())

        # 확정 로직 (기존과 동일)
        is_confirmed = False
        if not self.confirmed and pred != 0 and len(self.pred_history) > self.GRACE_PERIOD:
            if len(self.pred_history) >= self.CONFIRM_COUNT:
                recent = self.pred_history[-self.CONFIRM_COUNT:]
                if all(p == pred for p in recent):
                    self.confirmed = True
                    self.confirmed_label = LABELS[pred]
                    self.confirmed_idx = pred
                    self.confirm_sec = sec
                    is_confirmed = True

        return pred, prob, is_confirmed

    def get_final_label(self):
        """모든 sec 처리 후 최종 라벨 결정 (Late Correction 적용)."""
        if not self.confirmed:
            # 미확정이면 마지막 예측 사용
            if self.pred_history:
                return LABELS[self.pred_history[-1]]
            return "NORMAL"

        original_label = self.confirmed_label
        original_idx = self.confirmed_idx

        # Late Correction: 확정이 LOCA 계열일 때만 적용
        if original_idx not in LOCA_INDICES:
            return original_label

        # 마지막 late_window초의 예측 확인
        recent_preds = self.pred_history[-self.late_window:]
        recent_probs = self.prob_history[-self.late_window:]

        if len(recent_preds) < 3:  # 너무 적으면 교정 안 함
            return original_label

        # 최근 예측에서 가장 많은 클래스 (LOCA 계열만)
        loca_preds = [p for p in recent_preds if p in LOCA_INDICES]
        if not loca_preds:
            return original_label

        counter = Counter(loca_preds)
        most_common_idx, most_count = counter.most_common(1)[0]

        # 교정 조건:
        # 1) 최근 다수결이 확정과 다른 클래스
        # 2) 과반 이상이 같은 클래스
        # 3) 해당 클래스의 평균 confidence가 threshold 이상
        if most_common_idx == original_idx:
            return original_label  # 교정 불필요

        ratio = most_count / len(loca_preds)
        if ratio < 0.5:
            return original_label  # 과반 미달

        # 최근 확률에서 해당 클래스의 평균 confidence
        avg_conf = np.mean([p[most_common_idx] for p in recent_probs])
        if avg_conf < self.late_conf_threshold:
            return original_label  # confidence 부족

        corrected = LABELS[most_common_idx]
        return corrected


def run_test(model, scaler, feat, raw_names, late_window, late_conf, tag):
    """100건 테스트 실행."""
    answers = {}
    with open(ANSWERS_PATH, "r") as f:
        for row in csv.DictReader(f):
            answers[int(row["test_id"])] = row["label"]

    inf = InferenceWithLateCorrection(model, scaler, feat, raw_names,
                                       late_window=late_window,
                                       late_conf_threshold=late_conf)
    dummy = np.zeros((1, inf.WINDOW, len(feat.feature_names)), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    correct_orig = 0
    correct_late = 0
    corrections = []

    for tid in range(1, 101):
        folder = DATA_ROOT / f"test{tid}"
        if not folder.exists():
            continue

        sec_files = [p for p in folder.glob(f"test{tid}_sec*.csv") if " " not in p.name]
        sec_files = sorted(sec_files, key=lambda p: int(p.stem.split("sec")[1]))
        if not sec_files:
            continue

        inf.reset()
        header = None

        for sf in sec_files:
            sec = int(sf.stem.split("sec")[1])
            with open(sf, "r", encoding="utf-8", newline="") as f:
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
                    if not np.isfinite(val): val = 0.0
                except: val = 0.0
                values.append(val)
            x = np.array(values, dtype=np.float32)
            inf.process_sec(x, sec, header)

        # 원래 확정 라벨
        original = inf.confirmed_label or "NORMAL"
        # Late Correction 적용
        final = inf.get_final_label()

        ans = answers.get(tid, "?")
        if original == ans:
            correct_orig += 1
        if final == ans:
            correct_late += 1

        if original != final:
            ok_orig = "O" if original == ans else "X"
            ok_late = "O" if final == ans else "X"
            corrections.append((tid, ans, original, final, ok_orig, ok_late))

    print(f"\n  [{tag}] late_window={late_window}, conf={late_conf}")
    print(f"    원래: {correct_orig}/100  →  교정 후: {correct_late}/100")
    if corrections:
        print(f"    교정된 케이스:")
        for tid, ans, orig, final, ok_o, ok_l in corrections:
            print(f"      test{tid:>3d}: 정답={ans:>10s}  {orig:>10s}({ok_o}) -> {final:>10s}({ok_l})")

    return correct_orig, correct_late, corrections


def main():
    print("=" * 80)
    print("  Late Correction 후처리 테스트")
    print("=" * 80)

    # 5ep 모델
    prefix_5 = "tcn__feat=physics_v3__val=1__ep=5__cw=1__seed=0__skipd=5__dan=1__ss=5__win=3__stride=1"
    model_5 = tf.keras.models.load_model(str(MODEL_DIR / f"{prefix_5}__model.keras"), compile=False)
    scaler_5 = joblib.load(MODEL_DIR / f"{prefix_5}__scaler.pkl")
    feat_5 = joblib.load(MODEL_DIR / f"{prefix_5}__feature_transformer.pkl")
    raw_5 = feat_5.feature_names_all

    # 100ep 모델
    prefix_100 = "tcn__feat=physics_v3__val=1__ep=100__cw=1__seed=0__skipd=5__dan=1__ss=5__win=3__stride=1"
    model_100 = tf.keras.models.load_model(str(MODEL_DIR / f"{prefix_100}__model.keras"), compile=False)
    scaler_100 = joblib.load(MODEL_DIR / f"{prefix_100}__scaler.pkl")
    feat_100 = joblib.load(MODEL_DIR / f"{prefix_100}__feature_transformer.pkl")
    raw_100 = feat_100.feature_names_all

    print("\n" + "=" * 80)
    print("  5에폭 모델")
    print("=" * 80)
    for lw in [5, 10, 15, 20]:
        for lc in [0.5, 0.6, 0.7, 0.8]:
            run_test(model_5, scaler_5, feat_5, raw_5, lw, lc, f"5ep")

    print("\n" + "=" * 80)
    print("  100에폭 모델")
    print("=" * 80)
    for lw in [5, 10, 15, 20]:
        for lc in [0.5, 0.6, 0.7, 0.8]:
            run_test(model_100, scaler_100, feat_100, raw_100, lw, lc, f"100ep")


if __name__ == "__main__":
    main()
