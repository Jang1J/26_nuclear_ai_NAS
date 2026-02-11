"""
V3 학습 파이프라인.

v2(main.py)를 기반으로 3가지 개선 적용:
  A. Leak=1 sample weighting — leak=1 SGTR 샘플에 가중치 부여
  B. SGTR F1 기반 early stopping — val_loss 대신 SGTR F1으로 best 모델 선택
  C. Loop2 threshold 후처리 — inference 시 Loop2 과다예측 억제

v2 코드는 수정하지 않음. import하여 재사용.
"""
import re
import argparse
import time
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
import joblib
from tensorflow.keras import callbacks, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

# ── v2 코드 재사용 ──
from .dataloader import (
    load_Xy_runs,
    create_sliding_windows_from_runs,
    LABELS, ID2LABEL, LABEL2ID,
)
from .feature_method_v3 import make_feature_method_v3
from .feature_method_v4 import make_feature_method_v4
from .data_split import split_runs, compute_class_weight
from .model import (
    build_mlp, build_cnn, build_cnn_attention,
    build_tcn, build_transformer,
)
from .main import (
    focal_loss, WarmupCosineSchedule,
    _build_model_single, _compile_model,
    print_label_dist,
)
from .utils_plot import save_acc_loss, save_confusion_matrix, save_per_class_accuracy


# ═══════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════
def extract_leak_size(filename):
    """파일명에서 leak size 추출."""
    m = re.search(r'leak=(\d+)', filename)
    return int(m.group(1)) if m else -1


def create_sliding_windows_with_leak(X_runs, y_runs, leak_ids, window_size, stride=1, dilation=1):
    """
    슬라이딩 윈도우 생성 + leak size 추적.

    dilation: 윈도우 내부 간격. dilation=5이면 [행i, 행i+5, 행i+10] 형태.
              윈도우가 커버하는 실제 범위 = (window_size-1)*dilation + 1
              stride=1이면 1행(=실제 0.2초)씩 이동 → 데이터 dilation배 증가.

    Returns:
        X_windows, y_windows, leak_windows
    """
    all_X, all_y, all_leak = [], [], []
    span = (window_size - 1) * dilation + 1  # 윈도우가 필요한 총 행 수

    for X_run, y_run, leak_id in zip(X_runs, y_runs, leak_ids):
        N = X_run.shape[0]
        for i in range(0, N - span + 1, stride):
            indices = [i + j * dilation for j in range(window_size)]
            all_X.append(X_run[indices])
            all_y.append(y_run[indices[-1]])  # 마지막 타임스텝 라벨
            all_leak.append(leak_id)

    if len(all_X) == 0:
        raise ValueError("윈도우 생성 불가")

    return (
        np.array(all_X, dtype=np.float32),
        np.array(all_y, dtype=np.int64),
        np.array(all_leak, dtype=np.int32),
    )


def compute_sample_weights(y, leak_ids, leak1_weight=4.0):
    """
    Leak=1 SGTR 샘플에 가중치를 부여하는 sample_weight 생성.
    - SGTR(4,5,6) + leak=1 → leak1_weight
    - 그 외 → 1.0
    """
    weights = np.ones(len(y), dtype=np.float32)
    sgtr_mask = np.isin(y, [4, 5, 6])
    leak1_mask = (leak_ids == 1)
    weights[sgtr_mask & leak1_mask] = leak1_weight
    print(f"[Sample Weights] leak=1 SGTR: {np.sum(sgtr_mask & leak1_mask)} samples × {leak1_weight}")
    print(f"[Sample Weights] 나머지: {np.sum(~(sgtr_mask & leak1_mask))} samples × 1.0")
    return weights


def postprocess_loop2(y_prob, threshold=0.55):
    """
    Loop2 과다예측 억제 후처리.
    argmax가 Loop2(5)이고 confidence가 threshold 미만이면 2등 클래스로 교체.
    """
    y_pred = np.argmax(y_prob, axis=1).copy()
    n_changed = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 5:  # SGTR_Loop2
            if y_prob[i, 5] < threshold:
                probs = y_prob[i].copy()
                probs[5] = 0
                y_pred[i] = np.argmax(probs)
                n_changed += 1
    if n_changed > 0:
        print(f"[Loop2 후처리] {n_changed}개 샘플 Loop2 → 다른 클래스로 변경 (threshold={threshold})")
    return y_pred


# ═══════════════════════════════════════════════════════════
# SGTR F1 Callback
# ═══════════════════════════════════════════════════════════
class SGTRMetricCallback(callbacks.Callback):
    """
    매 epoch마다 validation set의 SGTR macro-F1을 계산하고
    best일 때 모델 가중치를 저장.
    """

    def __init__(self, X_val, y_val, save_path, patience=15):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.save_path = save_path
        self.patience = patience
        self.best_f1 = -1.0
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None
        self.sgtr_labels = [4, 5, 6]  # SGTR_Loop1, Loop2, Loop3

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        # SGTR 샘플만 추출
        sgtr_mask = np.isin(self.y_val, self.sgtr_labels)
        if sgtr_mask.sum() == 0:
            return

        y_true_sgtr = self.y_val[sgtr_mask]
        y_pred_sgtr = y_pred[sgtr_mask]

        # SGTR 3개 클래스에 대한 macro F1
        sgtr_f1 = f1_score(y_true_sgtr, y_pred_sgtr,
                           labels=self.sgtr_labels, average='macro', zero_division=0)

        print(f"  [SGTR F1] epoch {epoch+1}: {sgtr_f1:.4f} (best: {self.best_f1:.4f})")

        if sgtr_f1 > self.best_f1:
            self.best_f1 = sgtr_f1
            self.best_epoch = epoch
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            print(f"\n[SGTR F1] Best F1={self.best_f1:.4f} at epoch {self.best_epoch+1} → 복원")
            self.model.set_weights(self.best_weights)


# ═══════════════════════════════════════════════════════════
# V3 학습 파이프라인
# ═══════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", type=str, required=True)

    p.add_argument("--model_folder", type=str, default="models_9class_v3")
    p.add_argument("--train_folder", type=str, default="train_results_9class_v3")
    p.add_argument("--test_folder", type=str, default="test_results_9class_v3")

    p.add_argument(
        "--model_type", type=str, default="tcn",
        choices=["mlp", "cnn", "cnn_attention", "tcn", "transformer"],
    )

    p.add_argument("--use_val", action="store_true")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train", action="store_true")

    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--window_size", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)

    # 데이터 증강
    p.add_argument("--use_augmentation", action="store_true")
    p.add_argument("--jitter_std", type=float, default=0.01)
    p.add_argument("--scale_min", type=float, default=0.95)
    p.add_argument("--scale_max", type=float, default=1.05)
    p.add_argument("--augment_prob", type=float, default=0.5)

    p.add_argument("--augment_minority", action="store_true")
    p.add_argument("--minority_classes", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    p.add_argument("--minority_ratio", type=float, default=10.0)

    # 학습률 스케줄링
    p.add_argument("--lr_schedule", type=str, default=None,
                    choices=[None, "cosine", "warmup_cosine"])
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_min_factor", type=float, default=0.01)

    # 옵티마이저
    p.add_argument("--use_adamw", action="store_true")
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # 손실 함수
    p.add_argument("--use_focal_loss", action="store_true")
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # 콜백
    p.add_argument("--early_stopping_patience", type=int, default=25)

    # ── V3 전용 옵션 ──
    p.add_argument("--leak1_weight", type=float, default=4.0,
                    help="Leak=1 SGTR 샘플 가중치")
    p.add_argument("--loop2_threshold", type=float, default=0.55,
                    help="Loop2 후처리 confidence threshold")
    p.add_argument("--use_sgtr_callback", action="store_true",
                    help="SGTR F1 기반 best 모델 선택")
    p.add_argument("--skip_delay_rows", type=int, default=0,
                    help="사고 파일 앞 N행 제거 (delay 구간 오라벨 방지)")
    p.add_argument("--delay_as_normal", action="store_true",
                    help="사고 파일 delay 구간을 NORMAL로 재라벨링 (Operating Point 불일치 해결)")
    p.add_argument("--subsample_stride", type=int, default=1,
                    help="매 N행마다 1행만 사용 (시간간격 일치: 학습1초→대회5초=stride 5)")
    p.add_argument("--dilation", type=int, default=1,
                    help="윈도우 내부 간격 (dilation=5: [행i,행i+5,행i+10], 미분도 5행 간격)")
    p.add_argument("--feat_version", type=str, default="v3",
                    choices=["v3", "v4"],
                    help="피처 엔지니어링 버전 (v3=미분포함 266개, v4=미분제거 239개)")

    return p.parse_args()


def run_single_v3(args):
    is_seq_model = args["model_type"] in ("cnn", "cnn_attention", "tcn", "transformer")

    skip_d = args.get("skip_delay_rows", 0)
    dan = args.get("delay_as_normal", False)
    ss = args.get("subsample_stride", 1)
    dil = args.get("dilation", 1)
    feat_ver = args.get("feat_version", "v3")
    run_name = (
        f"{args['model_type']}"
        f"__feat=physics_{feat_ver}"
        f"__val={int(args['use_val'])}"
        f"__ep={args['epochs']}"
        f"__cw={int(args['use_class_weight'])}"
        f"__seed={args['seed']}"
        f"__skipd={skip_d}"
        f"__dan={int(dan)}"
        f"__ss={ss}"
        f"__dil={dil}"
    )
    if is_seq_model:
        run_name += f"__win={args['window_size']}__stride={args['stride']}"

    model_dir = Path(args["model_folder"])
    train_dir = Path(args["train_folder"]) / run_name
    test_dir = Path(args["test_folder"]) / run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{run_name}__model.keras"

    # ──────────────────────────────────────────
    # 1) 데이터 로드
    # ──────────────────────────────────────────
    X_runs, y_runs, run_names_list, feature_names = load_Xy_runs(
        args["data_folder"], include_time=False,
        skip_delay_rows=args.get("skip_delay_rows", 0),
        delay_as_normal=args.get("delay_as_normal", False),
        subsample_stride=args.get("subsample_stride", 1),
    )
    total_samples = sum(len(X) for X in X_runs)

    actual_classes = np.unique(np.concatenate(y_runs))
    n_classes = len(actual_classes)
    print(f"\n[V3 Data] {len(X_runs)} runs, {total_samples} samples, {len(feature_names)} features")
    print(f"[Classes] {n_classes} classes: {[ID2LABEL.get(int(c), c) for c in actual_classes]}")

    # Leak size 추출
    run_leak_sizes = [extract_leak_size(name) for name in run_names_list]
    print(f"[V3] Leak size 추출 완료: SGTR 파일 {sum(1 for ls in run_leak_sizes if ls > 0)}개")

    # ──────────────────────────────────────────
    # 2) 런 단위 split + leak 추적
    # ──────────────────────────────────────────
    # split_runs 내부 로직 재현하여 인덱스 추적
    rng = np.random.default_rng(args["seed"])
    run_labels = np.array([int(y[0]) for y in y_runs])

    train_idx, val_idx, test_idx = [], [], []
    for label in np.unique(run_labels):
        idx = np.where(run_labels == label)[0]
        rng.shuffle(idx)
        n = len(idx)

        if n == 1:
            train_idx.extend(idx)
            continue
        if n == 2:
            train_idx.append(idx[0])
            test_idx.append(idx[1])
            continue

        n_test = max(1, int(n * 0.2))
        if args["use_val"]:
            n_val = max(1, int(n * 0.1)) if n >= 5 else 0
            while n_test + n_val >= n and n_val > 0:
                n_val -= 1
            if n_test + n_val >= n:
                n_test = n - 1
                n_val = 0
        else:
            n_val = 0

        if n_test + n_val >= n:
            n_test = max(1, n - 1)
            n_val = 0

        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    rng.shuffle(np.array(train_idx))  # 단 인덱스만 참고용, 실제 데이터는 아래에서 추출

    train_X = [X_runs[i] for i in train_idx]
    train_y = [y_runs[i] for i in train_idx]
    train_leaks = [run_leak_sizes[i] for i in train_idx]

    test_X = [X_runs[i] for i in test_idx]
    test_y = [y_runs[i] for i in test_idx]
    test_leaks = [run_leak_sizes[i] for i in test_idx]

    val_X, val_y, val_leaks = [], [], []
    if args["use_val"]:
        val_X = [X_runs[i] for i in val_idx]
        val_y = [y_runs[i] for i in val_idx]
        val_leaks = [run_leak_sizes[i] for i in val_idx]

    print(f"\n[Split] train={len(train_X)} runs, val={len(val_X)} runs, test={len(test_X)} runs")

    # ──────────────────────────────────────────
    # 3) V3 Feature engineering
    # ──────────────────────────────────────────
    dilation = args.get("dilation", 1)
    if feat_ver == "v4":
        feat = make_feature_method_v4("physics_v4", moving_std_window=3)
        print("[V4] 미분 피처 제거 모드 — 시간 간격 무관")
    else:
        feat = make_feature_method_v3("physics_v3", moving_std_window=3, diff_stride=dilation)

    Xtr_flat = np.vstack(train_X)
    ytr_flat = np.concatenate(train_y)
    feat.fit(Xtr_flat, ytr_flat, feature_names)
    del Xtr_flat, ytr_flat

    train_X, train_y, feat_names = feat.transform_runs(train_X, train_y)
    test_X, test_y, _ = feat.transform_runs(test_X, test_y)
    if args["use_val"] and len(val_X) > 0:
        val_X, val_y, _ = feat.transform_runs(val_X, val_y)

    D = train_X[0].shape[1]
    print(f"\n[{feat_ver.upper()} Feature] physics_{feat_ver}: {len(feature_names)} -> {D} features")

    # ──────────────────────────────────────────
    # 4) Scaler
    # ──────────────────────────────────────────
    scaler = StandardScaler()
    Xtr_flat = np.vstack(train_X)
    scaler.fit(Xtr_flat)
    del Xtr_flat

    train_X = [scaler.transform(X) for X in train_X]
    test_X = [scaler.transform(X) for X in test_X]
    if args["use_val"] and len(val_X) > 0:
        val_X = [scaler.transform(X) for X in val_X]

    # ──────────────────────────────────────────
    # 5) 윈도우 생성 + leak 추적
    # ──────────────────────────────────────────
    W = args["window_size"]
    stride = args["stride"]

    if is_seq_model:
        Xtr_m, ytr, leak_tr = create_sliding_windows_with_leak(
            train_X, train_y, train_leaks, W, stride, dilation=dilation)
        Xte_m, yte, leak_te = create_sliding_windows_with_leak(
            test_X, test_y, test_leaks, W, stride, dilation=dilation)
        Xva_m, yva, leak_va = None, None, None
        if args["use_val"] and len(val_X) > 0:
            Xva_m, yva, leak_va = create_sliding_windows_with_leak(
                val_X, val_y, val_leaks, W, stride, dilation=dilation)
        print(f"\n[Window] size={W}, stride={stride}, dilation={dilation}")
        print(f"  train: {Xtr_m.shape}, test: {Xte_m.shape}", end="")
        if Xva_m is not None:
            print(f", val: {Xva_m.shape}")
        else:
            print()
    else:
        Xtr_m = np.vstack(train_X)
        ytr = np.concatenate(train_y)
        leak_tr = np.concatenate([np.full(len(y), ls, dtype=np.int32)
                                   for y, ls in zip(train_y, train_leaks)])
        Xte_m = np.vstack(test_X)
        yte = np.concatenate(test_y)
        leak_te = np.concatenate([np.full(len(y), ls, dtype=np.int32)
                                   for y, ls in zip(test_y, test_leaks)])
        Xva_m, yva, leak_va = None, None, None
        if args["use_val"] and len(val_X) > 0:
            Xva_m = np.vstack(val_X)
            yva = np.concatenate(val_y)
            leak_va = np.concatenate([np.full(len(y), ls, dtype=np.int32)
                                       for y, ls in zip(val_y, val_leaks)])

    print_label_dist("\ntrain (before aug)", ytr)

    # ──────────────────────────────────────────
    # 6) 데이터 증강
    # ──────────────────────────────────────────
    if args.get("use_augmentation") and args.get("train"):
        from .augmentation import TimeSeriesAugmenter, MinorityOversampler

        n_before = len(ytr)

        augmenter = TimeSeriesAugmenter(
            jitter_std=args.get("jitter_std", 0.01),
            scale_range=(args.get("scale_min", 0.95), args.get("scale_max", 1.05)),
            augment_prob=args.get("augment_prob", 0.5),
            seed=args["seed"],
        )
        Xtr_m, ytr = augmenter.augment(Xtr_m, ytr)
        # leak_tr도 확장 (증강된 샘플은 원본과 같은 leak)
        leak_tr_aug = np.concatenate([leak_tr, leak_tr])  # augment는 원본+증강
        leak_tr = leak_tr_aug[:len(ytr)]
        print(f"\n[Augmentation] Standard: {n_before} -> {len(ytr)} samples")

        if args.get("augment_minority"):
            minority_cls = args.get("minority_classes", [4, 5, 6, 7, 8])
            if minority_cls:
                n_before_min = len(ytr)
                minority = MinorityOversampler(
                    target_classes=minority_cls,
                    target_ratio=args.get("minority_ratio", 10.0),
                    jitter_std=args.get("jitter_std", 0.02),
                    scale_range=(args.get("scale_min", 0.90), args.get("scale_max", 1.10)),
                    seed=args["seed"],
                    balance_to_majority=True,
                )
                Xtr_m, ytr = minority.oversample(Xtr_m, ytr)
                # leak_tr 확장 (새로 추가된 샘플의 leak은 원본에서 추정 불가 → -1)
                n_added = len(ytr) - len(leak_tr)
                if n_added > 0:
                    leak_tr = np.concatenate([leak_tr, np.full(n_added, -1, dtype=np.int32)])
                print(f"[Augmentation] Minority: {n_before_min} -> {len(ytr)} samples")

        print_label_dist("\ntrain (after aug)", ytr)

    if yva is not None:
        print_label_dist("val", yva)
    print_label_dist("test", yte)

    # ──────────────────────────────────────────
    # 7) Sample weights (V3 핵심 A)
    # ──────────────────────────────────────────
    # Keras는 class_weight + sample_weight 동시 사용 불가
    # → class_weight를 sample_weight에 통합
    sample_weights = compute_sample_weights(ytr, leak_tr, leak1_weight=args.get("leak1_weight", 4.0))

    if args["use_class_weight"]:
        cw = compute_class_weight(ytr)
        print("\n[Class weights → sample_weight 통합]")
        for cid, w in sorted(cw.items()):
            print(f"  {ID2LABEL.get(cid, cid):15s}: {w:.3f}")
        # class_weight를 sample_weight에 곱하기
        for i in range(len(ytr)):
            sample_weights[i] *= cw.get(int(ytr[i]), 1.0)
        print(f"[Sample Weights] class_weight 통합 완료 (min={sample_weights.min():.3f}, max={sample_weights.max():.3f})")

    # ──────────────────────────────────────────
    # 8) 모델 빌드
    # ──────────────────────────────────────────
    model = _build_model_single(args, D=D, W=W if is_seq_model else None, n_classes=n_classes)
    print(f"\n[V3 Model] {args['model_type']}")
    model.summary()

    # ──────────────────────────────────────────
    # 9) 학습
    # ──────────────────────────────────────────
    history = None
    train_time = 0.0

    if args["train"]:
        print("\n[V3 Mode] TRAIN")

        steps_per_epoch = int(np.ceil(len(ytr) / args["batch_size"]))
        _compile_model(model, args, steps_per_epoch=steps_per_epoch)

        # 콜백 설정
        cb = []
        monitor_metric = "val_loss" if (args["use_val"] and Xva_m is not None) else "loss"

        if not args.get("lr_schedule"):
            cb.append(callbacks.ReduceLROnPlateau(
                monitor=monitor_metric, factor=0.5, patience=7, min_lr=1e-6, verbose=1,
            ))

        if args["use_val"] and Xva_m is not None:
            patience = args.get("early_stopping_patience", 25)

            # MLP 과적합 방지
            if args["model_type"] == "mlp" and patience > 10:
                patience = 10
                print(f"[MLP] early_stopping_patience → {patience}")

            mlp_min_delta = 0.0005 if args["model_type"] == "mlp" else 0.001
            cb.append(callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=patience,
                min_delta=mlp_min_delta,
                restore_best_weights=True, verbose=1,
            ))

            # V3 핵심 B: SGTR F1 기반 best 모델 선택
            if args.get("use_sgtr_callback", True):
                sgtr_cb = SGTRMetricCallback(
                    X_val=Xva_m, y_val=yva,
                    save_path=str(model_path),
                    patience=patience,
                )
                cb.append(sgtr_cb)
                print("[V3] SGTR F1 Callback 활성화")

        epochs = args["epochs"]

        fit_kwargs = dict(
            epochs=epochs,
            batch_size=args["batch_size"],
            callbacks=cb,
            verbose=1,
            sample_weight=sample_weights,  # V3: leak=1 + class_weight 통합
        )
        if args["use_val"] and Xva_m is not None:
            fit_kwargs["validation_data"] = (Xva_m, yva)

        t0 = time.time()
        history = model.fit(Xtr_m, ytr, **fit_kwargs)
        train_time = time.time() - t0
        print(f"\n[Training time] {train_time:.1f}s")

        model.save(model_path)
        print("[Saved model]", model_path)

        # 전처리 저장
        scaler_path = model_dir / f"{run_name}__scaler.pkl"
        joblib.dump(scaler, scaler_path)

        feat_path = model_dir / f"{run_name}__feature_transformer.pkl"
        joblib.dump(feat, feat_path)

        config = {
            "version": feat_ver,
            "model_type": args["model_type"],
            "feature_method": f"physics_{feat_ver}",
            "window_size": args["window_size"],
            "stride": args["stride"],
            "epochs": args["epochs"],
            "actual_epochs": epochs,
            "batch_size": args["batch_size"],
            "lr": args["lr"],
            "seed": args["seed"],
            "leak1_weight": args.get("leak1_weight", 4.0),
            "loop2_threshold": args.get("loop2_threshold", 0.55),
            "use_sgtr_callback": args.get("use_sgtr_callback", True),
            "n_features": D,
        }
        config_path = model_dir / f"{run_name}__config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # 클래스 매핑
        available_classes = np.unique(ytr)
        class_mapping_path = model_dir / f"{run_name}__class_mapping.npy"
        np.save(class_mapping_path, available_classes)

        if history is not None:
            save_acc_loss(history, str(train_dir / "acc_vs_epoch.png"),
                          str(train_dir / "loss_vs_epoch.png"))

    else:
        print("\n[V3 Mode] LOAD")
        model = tf.keras.models.load_model(model_path)

    # ──────────────────────────────────────────
    # 10) 테스트
    # ──────────────────────────────────────────
    test_loss, test_acc = model.evaluate(Xte_m, yte, verbose=0)

    t0 = time.time()
    y_prob = model.predict(Xte_m, verbose=0)
    infer_time = time.time() - t0

    # V3 핵심 C: Loop2 후처리
    y_pred_raw = np.argmax(y_prob, axis=1)
    y_pred_pp = postprocess_loop2(y_prob, threshold=args.get("loop2_threshold", 0.55))

    n_test = len(yte)
    per_sample_ms = (infer_time / n_test) * 1000

    unique_classes = np.unique(yte)
    target_names = [LABELS[i] for i in unique_classes]

    # 후처리 전/후 비교
    print(f"\n[Test result - 후처리 전]")
    print(f"  loss: {test_loss:.6f}, acc: {test_acc:.6f}")
    print(classification_report(yte, y_pred_raw, labels=unique_classes, target_names=target_names))

    acc_pp = np.mean(y_pred_pp == yte)
    print(f"[Test result - Loop2 후처리 후]")
    print(f"  acc: {acc_pp:.6f}")
    print(classification_report(yte, y_pred_pp, labels=unique_classes, target_names=target_names))

    # 메트릭 저장 (후처리 후 기준)
    metrics_path = test_dir / "test_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_acc : {test_acc:.6f}\n")
        f.write(f"test_acc_postprocess: {acc_pp:.6f}\n")
        f.write(f"inference_time: {infer_time:.3f}s ({per_sample_ms:.4f}ms/sample)\n")
        if args["train"]:
            f.write(f"train_time: {train_time:.1f}s\n")
        f.write(f"\n--- 후처리 전 ---\n")
        f.write(classification_report(yte, y_pred_raw, labels=unique_classes, target_names=target_names))
        f.write(f"\n--- Loop2 후처리 후 (threshold={args.get('loop2_threshold', 0.55)}) ---\n")
        f.write(classification_report(yte, y_pred_pp, labels=unique_classes, target_names=target_names))

    # 시각화
    cm_path = test_dir / "confusion_matrix.png"
    per_class_path = test_dir / "per_class_accuracy.png"
    save_confusion_matrix(yte, y_pred_pp, LABELS, str(cm_path), available_classes=unique_classes)
    save_per_class_accuracy(yte, y_pred_pp, LABELS, str(per_class_path), available_classes=unique_classes)
    print("[Saved test results]", test_dir)


def main():
    args = vars(parse_args())
    np.random.seed(args["seed"])
    tf.random.set_seed(args["seed"])
    run_single_v3(args)


if __name__ == "__main__":
    main()
