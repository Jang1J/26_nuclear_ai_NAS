"""
누수 없는 학습 파이프라인.

순서:
  1. 런(파일) 단위 데이터 로드
  2. 런 단위로 train/val/test 분할 (같은 런이 여러 세트에 안 섞임)
  3. train 런에서만 feature_method.fit → 모든 세트에 transform_runs
  4. train에서만 scaler.fit → 모든 세트에 transform
  5. 슬라이딩 윈도우 생성 (런 경계 보존, stride 설정 가능)
  5.5. (옵션) 데이터 증강 적용
  6. 모델 학습/평가

v2: 데이터 증강, 고급 LR 스케줄링, AdamW, Focal Loss 추가
"""
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
from sklearn.metrics import classification_report

from .dataloader import (
    load_Xy_runs,
    create_sliding_windows_from_runs,
    LABELS, ID2LABEL,
)
from .feature_method import make_feature_method
from .data_split import split_runs, compute_class_weight
from .model import (
    build_mlp,
    build_cnn,
    build_cnn_attention,
    build_tcn,
    build_transformer,
)
from .utils_plot import save_acc_loss, save_confusion_matrix, save_per_class_accuracy


def print_label_dist(name, y):
    labels, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"{name} label distribution:")
    for l, c in zip(labels, counts):
        print(f"  {ID2LABEL.get(int(l), f'UNKNOWN({l})'):15s}: {c:6d} ({c/total:.3f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", type=str, required=True)

    p.add_argument("--model_folder", type=str, default="models")
    p.add_argument("--train_folder", type=str, default="train_results")
    p.add_argument("--test_folder", type=str, default="test_results")

    p.add_argument(
        "--model_type", type=str, default="cnn",
        choices=["mlp", "cnn", "cnn_attention", "tcn", "transformer"],
    )

    p.add_argument("--use_val", action="store_true")

    # 학습 설정
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--train", action="store_true")

    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--window_size", type=int, default=3, help="윈도우 크기 (3=3초 @1초 샘플링)")
    p.add_argument("--stride", type=int, default=1, help="윈도우 이동 간격 (1=1초 @1초 샘플링)")

    # ── 데이터 증강 ──
    p.add_argument("--use_augmentation", action="store_true",
                    help="전체 클래스 데이터 증강 활성화")
    p.add_argument("--jitter_std", type=float, default=0.01,
                    help="Jitter 노이즈 표준편차 (신호 std 대비)")
    p.add_argument("--scale_min", type=float, default=0.95,
                    help="스케일링 최솟값")
    p.add_argument("--scale_max", type=float, default=1.05,
                    help="스케일링 최댓값")
    p.add_argument("--augment_prob", type=float, default=0.5,
                    help="증강 적용 확률")

    # ── 소수 클래스 증강 ──
    p.add_argument("--augment_minority", action="store_true",
                    help="소수 클래스 오버샘플링 활성화")
    p.add_argument("--minority_classes", type=int, nargs="+", default=[2, 8],
                    help="증강할 소수 클래스 ID (기본: 2=LOCA_CL, 8=ESDE_out)")
    p.add_argument("--minority_ratio", type=float, default=2.0,
                    help="소수 클래스 증강 비율")

    # ── 학습률 스케줄링 ──
    p.add_argument("--lr_schedule", type=str, default=None,
                    choices=[None, "cosine", "warmup_cosine"],
                    help="학습률 스케줄 (None=ReduceLROnPlateau)")
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="Warmup 비율 (warmup_cosine만)")
    p.add_argument("--lr_min_factor", type=float, default=0.01,
                    help="최소 학습률 = lr × lr_min_factor")

    # ── 옵티마이저 ──
    p.add_argument("--use_adamw", action="store_true",
                    help="AdamW 사용 (weight decay 정규화)")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                    help="Weight decay 계수")

    # ── 손실 함수 ──
    p.add_argument("--use_focal_loss", action="store_true",
                    help="Focal Loss 사용 (클래스 불균형 대응)")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal Loss γ 파라미터")

    # ── 콜백 ──
    p.add_argument("--early_stopping_patience", type=int, default=15,
                    help="EarlyStopping patience")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════
def focal_loss(gamma=2.0):
    """
    Focal Loss: 쉬운 샘플에 낮은 가중치, 어려운 샘플에 높은 가중치.
    FL(p_t) = -(1 - p_t)^γ × log(p_t)
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        n_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, n_classes)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        ce = -tf.math.log(p_t)

        return tf.reduce_mean(focal_weight * ce)
    return loss_fn


# ═══════════════════════════════════════════════════════════
# Warmup + Cosine LR Schedule
# ═══════════════════════════════════════════════════════════
@keras.saving.register_keras_serializable(package="NAS")
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup 후 Cosine Decay."""

    def __init__(self, initial_lr, total_steps, warmup_steps, min_lr_factor=0.01):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = initial_lr * min_lr_factor

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Warmup 단계
        warmup_lr = self.initial_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay 단계
        cos_step = step - warmup_steps
        cos_total = total_steps - warmup_steps
        cos_decay = 0.5 * (1.0 + tf.cos(
            np.pi * tf.minimum(cos_step / tf.maximum(cos_total, 1.0), 1.0)
        ))
        cosine_lr = self.min_lr + (self.initial_lr - self.min_lr) * cos_decay

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_factor": self.min_lr / self.initial_lr if self.initial_lr > 0 else 0.01,
        }


def _build_feature_engineer(args, model_dir: Path, train_dir: Path):
    return make_feature_method("physics", moving_std_window=3)


def _build_model_single(args, D, W=None):
    n_classes = len(LABELS)
    model_type = args["model_type"]

    if model_type == "mlp":
        return build_mlp(D, n_classes)

    if W is None:
        raise ValueError(f"{model_type} requires window_size")

    if model_type == "cnn":
        return build_cnn(W, D, n_classes)
    if model_type == "cnn_attention":
        return build_cnn_attention(W, D, n_classes)
    if model_type == "tcn":
        return build_tcn(W, D, n_classes)
    if model_type == "transformer":
        return build_transformer(W, D, n_classes)

    raise ValueError(f"Unknown model_type: {model_type}")


def _compile_model(model, args, steps_per_epoch=None):
    """모델 컴파일 (옵티마이저, 손실 함수 설정)."""

    # ── 학습률 스케줄 ──
    lr = args["lr"]
    if args.get("lr_schedule") and steps_per_epoch:
        total_steps = args["epochs"] * steps_per_epoch

        if args["lr_schedule"] == "cosine":
            lr = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args["lr"],
                decay_steps=total_steps,
                alpha=args.get("lr_min_factor", 0.01),
            )
        elif args["lr_schedule"] == "warmup_cosine":
            warmup_steps = int(total_steps * args.get("warmup_ratio", 0.1))
            lr = WarmupCosineSchedule(
                initial_lr=args["lr"],
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_factor=args.get("lr_min_factor", 0.01),
            )

    # ── 옵티마이저 ──
    if args.get("use_adamw"):
        optimizer = optimizers.AdamW(
            learning_rate=lr,
            weight_decay=args.get("weight_decay", 1e-4),
        )
    else:
        optimizer = optimizers.Adam(learning_rate=lr)

    # ── 손실 함수 ──
    if args.get("use_focal_loss"):
        loss = focal_loss(gamma=args.get("focal_gamma", 2.0))
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
        )

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


def _fit_model(model, Xtr, ytr, Xva, yva, args, class_weight=None):
    cb = []
    monitor_metric = "val_loss" if (args["use_val"] and Xva is not None) else "loss"

    # ReduceLROnPlateau: LR 스케줄이 없을 때만 사용
    if not args.get("lr_schedule"):
        cb.append(
            callbacks.ReduceLROnPlateau(
                monitor=monitor_metric, factor=0.5, patience=7, min_lr=1e-6, verbose=1,
            )
        )

    if args["use_val"] and Xva is not None:
        patience = args.get("early_stopping_patience", 15)
        cb.append(
            callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=patience,
                restore_best_weights=True, verbose=1,
            )
        )

    fit_kwargs = dict(
        epochs=args["epochs"],
        batch_size=args["batch_size"],
        callbacks=cb,
        verbose=1,
    )
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight
    if args["use_val"] and Xva is not None:
        fit_kwargs["validation_data"] = (Xva, yva)

    t0 = time.time()
    history = model.fit(Xtr, ytr, **fit_kwargs)
    train_time = time.time() - t0
    return history, train_time


def run_single(args):
    is_seq_model = args["model_type"] in ("cnn", "cnn_attention", "tcn", "transformer")

    run_name = (
        f"{args['model_type']}"
        f"__feat=physics"
        f"__val={int(args['use_val'])}"
        f"__ep={args['epochs']}"
        f"__cw={int(args['use_class_weight'])}"
        f"__seed={args['seed']}"
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

    # ──────────────────────────────────────────────
    # 1) 런 단위 데이터 로드
    # ──────────────────────────────────────────────
    X_runs, y_runs, run_names_list, feature_names = load_Xy_runs(
        args["data_folder"], include_time=False
    )
    total_samples = sum(len(X) for X in X_runs)
    print(f"\n[Data] {len(X_runs)} runs, {total_samples} samples, {len(feature_names)} features")

    # ──────────────────────────────────────────────
    # 2) 런 단위 train/val/test 분할 (누수 방지)
    # ──────────────────────────────────────────────
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = split_runs(
        X_runs, y_runs, run_names_list,
        val_ratio=0.1, test_ratio=0.2,
        seed=args["seed"], use_val=args["use_val"],
    )
    print(f"\n[Split] train={len(train_X)} runs, val={len(val_X)} runs, test={len(test_X)} runs")

    # ──────────────────────────────────────────────
    # 3) Feature engineering: train에서만 fit, 모든 세트에 transform_runs
    # ──────────────────────────────────────────────
    feat = _build_feature_engineer(args, model_dir, train_dir)

    # train 합쳐서 fit (feature selection 등에 필요)
    Xtr_flat = np.vstack(train_X)
    ytr_flat = np.concatenate(train_y)
    feat.fit(Xtr_flat, ytr_flat, feature_names)
    del Xtr_flat, ytr_flat  # 메모리 해제

    # 런 단위로 transform (diff/stats/physics가 런 경계 내에서만 계산됨)
    train_X, train_y, feat_names = feat.transform_runs(train_X, train_y)
    test_X, test_y, _ = feat.transform_runs(test_X, test_y)
    if args["use_val"] and len(val_X) > 0:
        val_X, val_y, _ = feat.transform_runs(val_X, val_y)

    D = train_X[0].shape[1]
    print(f"\n[Feature] physics: {len(feature_names)} -> {D} features")

    # ──────────────────────────────────────────────
    # 4) Scaler: train에서만 fit, 모든 세트에 transform
    # ──────────────────────────────────────────────
    scaler = StandardScaler()
    Xtr_flat = np.vstack(train_X)
    scaler.fit(Xtr_flat)
    del Xtr_flat

    train_X = [scaler.transform(X) for X in train_X]
    test_X = [scaler.transform(X) for X in test_X]
    if args["use_val"] and len(val_X) > 0:
        val_X = [scaler.transform(X) for X in val_X]

    # ──────────────────────────────────────────────
    # 5) 윈도우 생성 또는 flat 데이터 준비
    # ──────────────────────────────────────────────
    W = args["window_size"]
    stride = args["stride"]

    if is_seq_model:
        Xtr_m, ytr = create_sliding_windows_from_runs(train_X, train_y, W, stride)
        Xte_m, yte = create_sliding_windows_from_runs(test_X, test_y, W, stride)
        Xva_m, yva = None, None
        if args["use_val"] and len(val_X) > 0:
            Xva_m, yva = create_sliding_windows_from_runs(val_X, val_y, W, stride)
        print(f"\n[Window] size={W}, stride={stride}")
        print(f"  train: {Xtr_m.shape}, test: {Xte_m.shape}", end="")
        if Xva_m is not None:
            print(f", val: {Xva_m.shape}")
        else:
            print()
    else:
        # MLP: 각 런을 flat으로 합침
        Xtr_m = np.vstack(train_X)
        ytr = np.concatenate(train_y)
        Xte_m = np.vstack(test_X)
        yte = np.concatenate(test_y)
        Xva_m, yva = None, None
        if args["use_val"] and len(val_X) > 0:
            Xva_m = np.vstack(val_X)
            yva = np.concatenate(val_y)

    print_label_dist("\ntrain (before aug)", ytr)

    # ──────────────────────────────────────────────
    # 5.5) 데이터 증강 (학습 데이터에만 적용)
    # ──────────────────────────────────────────────
    if args.get("use_augmentation") and args.get("train"):
        from .augmentation import TimeSeriesAugmenter, MinorityOversampler

        n_before = len(ytr)

        # 전체 클래스 증강
        augmenter = TimeSeriesAugmenter(
            jitter_std=args.get("jitter_std", 0.01),
            scale_range=(args.get("scale_min", 0.95), args.get("scale_max", 1.05)),
            augment_prob=args.get("augment_prob", 0.5),
            seed=args["seed"],
        )
        Xtr_m, ytr = augmenter.augment(Xtr_m, ytr)
        print(f"\n[Augmentation] Standard: {n_before} -> {len(ytr)} samples")

        # 소수 클래스 오버샘플링
        if args.get("augment_minority"):
            n_before_min = len(ytr)
            minority = MinorityOversampler(
                target_classes=args.get("minority_classes", [8]),
                target_ratio=args.get("minority_ratio", 2.0),
                jitter_std=0.02,
                scale_range=(0.90, 1.10),
                seed=args["seed"],
            )
            Xtr_m, ytr = minority.oversample(Xtr_m, ytr)
            print(f"[Augmentation] Minority: {n_before_min} -> {len(ytr)} samples")

        print_label_dist("\ntrain (after aug)", ytr)

    if yva is not None:
        print_label_dist("val", yva)
    print_label_dist("test", yte)

    # 6) class_weight
    class_weight = None
    if args["use_class_weight"]:
        class_weight = compute_class_weight(ytr)
        print("\n[Class weights]")
        for cid, w in sorted(class_weight.items()):
            print(f"  {ID2LABEL.get(cid, cid):15s}: {w:.3f}")

    # 7) 모델 빌드
    model = _build_model_single(args, D=D, W=W if is_seq_model else None)
    print(f"\n[Model] {args['model_type']}")
    model.summary()

    # 8) 학습 or 로드
    history = None
    train_time = 0.0

    if args["train"]:
        print("\n[Mode] TRAIN")

        # steps_per_epoch 계산 (LR 스케줄용)
        steps_per_epoch = int(np.ceil(len(ytr) / args["batch_size"]))
        _compile_model(model, args, steps_per_epoch=steps_per_epoch)

        # 학습 설정 로깅
        if args.get("lr_schedule"):
            print(f"  LR schedule: {args['lr_schedule']}")
        if args.get("use_adamw"):
            print(f"  Optimizer: AdamW (wd={args.get('weight_decay', 1e-4)})")
        if args.get("use_focal_loss"):
            print(f"  Loss: Focal Loss (γ={args.get('focal_gamma', 2.0)})")

        history, train_time = _fit_model(model, Xtr_m, ytr, Xva_m, yva, args, class_weight=class_weight)
        print(f"\n[Training time] {train_time:.1f}s")

        model.save(model_path)
        print("[Saved model]", model_path)

        # 전처리 파이프라인 저장
        from .preprocessing import PreprocessingPipeline

        pipeline = PreprocessingPipeline(feature_method="physics")
        pipeline.feature_transformer = feat
        pipeline.scaler = scaler
        pipeline.feature_names_in = feature_names
        pipeline.feature_names_out = feat_names
        pipeline._is_fitted = True
        pipeline.save(model_dir, prefix=run_name)

        # 개별 파일도 저장 (하위 호환성)
        scaler_path = model_dir / f"{run_name}__scaler.pkl"
        joblib.dump(scaler, scaler_path)

        # Config 저장 (모든 하이퍼파라미터 포함)
        config = {
            "model_type": args["model_type"],
            "feature_method": "physics",
            "window_size": args["window_size"],
            "stride": args["stride"],
            "epochs": args["epochs"],
            "batch_size": args["batch_size"],
            "lr": args["lr"],
            "seed": args["seed"],
            "use_val": args["use_val"],
            "use_class_weight": args["use_class_weight"],
            "data_folder": args["data_folder"],
            "use_augmentation": args.get("use_augmentation", False),
            "augment_minority": args.get("augment_minority", False),
            "lr_schedule": args.get("lr_schedule"),
            "use_adamw": args.get("use_adamw", False),
            "weight_decay": args.get("weight_decay", 1e-4),
            "use_focal_loss": args.get("use_focal_loss", False),
            "focal_gamma": args.get("focal_gamma", 2.0),
            "early_stopping_patience": args.get("early_stopping_patience", 15),
        }
        config_path = model_dir / f"{run_name}__config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[Saved config] {config_path}")

        # 클래스 매핑 저장
        available_classes = np.unique(ytr)
        class_mapping_path = model_dir / f"{run_name}__class_mapping.npy"
        np.save(class_mapping_path, available_classes)
        print(f"[Saved class mapping] {[ID2LABEL[int(c)] for c in available_classes]}")

        if history is not None:
            train_acc_path = train_dir / "acc_vs_epoch.png"
            train_loss_path = train_dir / "loss_vs_epoch.png"
            save_acc_loss(history, str(train_acc_path), str(train_loss_path))
            print("[Saved train curves]")

    else:
        print("\n[Mode] LOAD")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("[Loaded model]", model_path)

    # 9) 테스트
    test_loss, test_acc = model.evaluate(Xte_m, yte, verbose=0)

    t0 = time.time()
    y_prob = model.predict(Xte_m, verbose=0)
    infer_time = time.time() - t0
    y_pred = np.argmax(y_prob, axis=1)

    n_test = len(yte)
    per_sample_ms = (infer_time / n_test) * 1000

    unique_classes = np.unique(yte)
    target_names = [LABELS[i] for i in unique_classes]

    metrics_path = test_dir / "test_metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_acc : {test_acc:.6f}\n")
        f.write(f"inference_time: {infer_time:.3f}s ({per_sample_ms:.4f}ms/sample)\n")
        if args["train"]:
            f.write(f"train_time: {train_time:.1f}s\n")
        f.write(f"\n{classification_report(yte, y_pred, labels=unique_classes, target_names=target_names)}\n")

    print(f"\n[Test result]")
    print(f"  loss: {test_loss:.6f}")
    print(f"  acc : {test_acc:.6f}")
    print(f"  inference: {infer_time:.3f}s total, {per_sample_ms:.4f}ms/sample")
    print(f"\n{classification_report(yte, y_pred, labels=unique_classes, target_names=target_names)}")

    # 10) 시각화
    cm_path = test_dir / "confusion_matrix.png"
    per_class_path = test_dir / "per_class_accuracy.png"
    save_confusion_matrix(yte, y_pred, LABELS, str(cm_path), available_classes=unique_classes)
    save_per_class_accuracy(yte, y_pred, LABELS, str(per_class_path), available_classes=unique_classes)
    print("[Saved test results]", test_dir)


def main():
    args = vars(parse_args())
    np.random.seed(args["seed"])
    tf.random.set_seed(args["seed"])
    run_single(args)


if __name__ == "__main__":
    main()
