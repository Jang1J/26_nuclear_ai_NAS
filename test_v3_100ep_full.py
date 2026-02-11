"""V3 100ep 모델 전체 테스트."""
import os, sys, csv, warnings
from pathlib import Path
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

NAS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NAS_DIR / "Team N code" / "py"))

# practice 모듈
sys.path.insert(0, str(NAS_DIR / "Team N code" / "py"))

LABELS = [
    "NORMAL", "LOCA_HL", "LOCA_CL", "LOCA_RCP",
    "SGTR_Loop1", "SGTR_Loop2", "SGTR_Loop3",
    "ESDE_in", "ESDE_out",
]

MODEL_DIR = NAS_DIR / "models_9class_v3_ss5"
PREFIX = "tcn__feat=physics_v3__val=1__ep=100__cw=1__seed=0__skipd=5__dan=1__ss=5__win=3__stride=1"
DATA_ROOT = NAS_DIR / "Team N code" / "data"
ANSWERS_PATH = NAS_DIR / "data" / "real_test_data" / "answers.csv"


def sanitize_array(arr):
    return np.where(np.isfinite(arr), arr, 0.0)


class RealtimeInference100:
    WINDOW = 3
    CONFIRM_COUNT = 3
    GRACE_PERIOD = 5

    def __init__(self, model, scaler, feat, raw_names):
        self.model = model
        self.scaler = scaler
        self.feat = feat
        self.raw_feature_names = raw_names
        self._cached_header = None
        self._cached_col_to_idx = None
        self.reset()

    def reset(self):
        self.buffer = []
        self.pred_history = []
        self.confirmed = False
        self.confirmed_label = None

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
            return pred, prob, False

        self.pred_history.append(pred)
        is_confirmed = False
        if not self.confirmed and pred != 0 and len(self.pred_history) > self.GRACE_PERIOD:
            if len(self.pred_history) >= self.CONFIRM_COUNT:
                recent = self.pred_history[-self.CONFIRM_COUNT:]
                if all(p == pred for p in recent):
                    self.confirmed = True
                    self.confirmed_label = LABELS[pred]
                    is_confirmed = True
        return pred, prob, is_confirmed


def main():
    answers = {}
    with open(ANSWERS_PATH, "r") as f:
        for row in csv.DictReader(f):
            answers[int(row["test_id"])] = row["label"]

    model = tf.keras.models.load_model(str(MODEL_DIR / f"{PREFIX}__model.keras"), compile=False)
    scaler = joblib.load(MODEL_DIR / f"{PREFIX}__scaler.pkl")
    feat = joblib.load(MODEL_DIR / f"{PREFIX}__feature_transformer.pkl")
    raw_names = feat.feature_names_all
    print(f"[100ep] Features: {len(raw_names)} -> {len(feat.feature_names)}")

    inf = RealtimeInference100(model, scaler, feat, raw_names)
    dummy = np.zeros((1, inf.WINDOW, len(feat.feature_names)), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)

    correct = 0
    errors = []
    total = 0

    for tid in range(1, 101):  # 1~100만 유효
        folder = DATA_ROOT / f"test{tid}"
        if not folder.exists():
            continue
        total += 1

        sec_files = [p for p in folder.glob(f"test{tid}_sec*.csv") if " " not in p.name]
        sec_files = sorted(sec_files, key=lambda p: int(p.stem.split("sec")[1]))
        if not sec_files:
            continue

        inf.reset()
        header = None
        confirm_sec = None
        last_prob = None
        last_pred = 0

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
            pred, prob, is_confirmed = inf.process_sec(x, sec, header)
            last_pred = pred
            last_prob = prob
            if is_confirmed and confirm_sec is None:
                confirm_sec = sec

        confirmed_label = inf.confirmed_label or LABELS[last_pred]
        ans = answers.get(tid, "?")
        if confirmed_label == ans:
            correct += 1
        else:
            conf = last_prob[last_pred] if last_prob is not None else 0
            top2_idx = np.argsort(last_prob)[::-1][:3] if last_prob is not None else [0, 0, 0]
            top3 = [(LABELS[i], last_prob[i]) for i in top2_idx]
            errors.append({"tid": tid, "ans": ans, "pred": confirmed_label,
                          "conf": conf, "sec": confirm_sec, "top3": top3})

    print(f"\n{'='*80}")
    print(f"  V3 100ep: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"{'='*80}")

    class_c, class_t = defaultdict(int), defaultdict(int)
    err_tids = {e["tid"] for e in errors}
    for tid in range(1, 101):
        ans = answers.get(tid, "?")
        folder = DATA_ROOT / f"test{tid}"
        if not folder.exists(): continue
        class_t[ans] += 1
        if tid not in err_tids:
            class_c[ans] += 1

    print(f"\n  [클래스별]")
    for label in LABELS:
        t = class_t.get(label, 0)
        c = class_c.get(label, 0)
        if t > 0:
            print(f"    {label:>12s}: {c}/{t} ({c/t*100:.1f}%)")

    if errors:
        print(f"\n  [오답] ({len(errors)}건)")
        for e in errors:
            sec_str = f"sec{e['sec']}" if e['sec'] else "---"
            print(f"    test{e['tid']:>3d}: {e['ans']:>12s} -> {e['pred']:>12s}  conf={e['conf']:.4f}  {sec_str}")
            t = e["top3"]
            print(f"             top3: {t[0][0]}({t[0][1]:.4f}), {t[1][0]}({t[1][1]:.4f}), {t[2][0]}({t[2][1]:.4f})")


if __name__ == "__main__":
    main()
