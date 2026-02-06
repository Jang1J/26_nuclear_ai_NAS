"""
누수 없는 학습 파이프라인.

순서:
  1. 런(파일) 단위 데이터 로드
  2. 런 단위로 train/val/test 분할 (같은 런이 여러 세트에 안 섞임)
  3. train 런에서만 feature_method.fit → 모든 세트에 transform_runs
  4. train에서만 scaler.fit → 모든 세트에 transform
  5. 슬라이딩 윈도우 생성 (런 경계 보존, stride 설정 가능)
  6. 모델 학습/평가
"""
import argparse
import time
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
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
    build_lstm,
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
        choices=["mlp", "cnn", "cnn_attention", "lstm", "transformer"],
    )

    p.add_argument(
        "--feature_method", type=str, default="all",
        choices=["all", "change", "selection", "diff", "stats", "physics"],
    )

    p.add_argument("--use_val", action="store_true")

    # 학습 설정
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--train", action="store_true")
    p.add_argument("--test_noise", type=float, default=0.0)

    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--window_size", type=int, default=3, help="윈도우 크기 (3=3초 @1초 샘플링)")
    p.add_argument("--stride", type=int, default=1, help="윈도우 이동 간격 (1=1초 @1초 샘플링)")

    # selection / stats 옵션
    p.add_argument("--topk", type=int, default=300)
    p.add_argument("--stat_window", type=int, default=5)

    return p.parse_args()


def _build_feature_engineer(args, model_dir: Path, train_dir: Path):
    if args["feature_method"] == "selection":
        return make_feature_method(
            "selection",
            seed=args["seed"],
            model_path=str(model_dir / "feature_selector_lgbm.pkl"),
            save_model=True,
            importance_type="split",
            topk=args["topk"],
            topk_plot_path=str(train_dir / "top20_importance.png"),
        )
    if args["feature_method"] == "stats":
        return make_feature_method("stats", stat_window=args["stat_window"])
    if args["feature_method"] == "physics":
        return make_feature_method("physics", moving_std_window=3)
    return make_feature_method(args["feature_method"])


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
    if model_type == "lstm":
        return build_lstm(W, D, n_classes)
    if model_type == "transformer":
        return build_transformer(W, D, n_classes)

    raise ValueError(f"Unknown model_type: {model_type}")


def _compile_model(model, args):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )


def _fit_model(model, Xtr, ytr, Xva, yva, args, class_weight=None):
    cb = []
    monitor_metric = "val_loss" if (args["use_val"] and Xva is not None) else "loss"
    cb.append(
        callbacks.ReduceLROnPlateau(
            monitor=monitor_metric, factor=0.5, patience=7, min_lr=1e-6, verbose=1,
        )
    )
    if args["use_val"] and Xva is not None:
        cb.append(
            callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1,
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
    is_seq_model = args["model_type"] in ("cnn", "cnn_attention", "lstm", "transformer")

    run_name = (
        f"{args['model_type']}"
        f"__feat={args['feature_method']}"
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
    print(f"\n[Feature] {args['feature_method']}: {len(feature_names)} -> {D} features")

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

    print_label_dist("\ntrain", ytr)
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
        _compile_model(model, args)
        history, train_time = _fit_model(model, Xtr_m, ytr, Xva_m, yva, args, class_weight=class_weight)
        print(f"\n[Training time] {train_time:.1f}s")

        model.save(model_path)
        print("[Saved model]", model_path)

        # 전처리 파이프라인 저장
        import joblib
        from .preprocessing import PreprocessingPipeline

        pipeline = PreprocessingPipeline(
            feature_method=args["feature_method"],
            **{k: args[k] for k in ["topk", "stat_window"] if k in args}
        )
        pipeline.feature_transformer = feat
        pipeline.scaler = scaler
        pipeline.feature_names_in = feature_names
        pipeline.feature_names_out = feat_names
        pipeline._is_fitted = True
        pipeline.save(model_dir)

        # 개별 파일도 저장 (하위 호환성)
        scaler_path = model_dir / f"{run_name}__scaler.pkl"
        joblib.dump(scaler, scaler_path)

        # Config 저장
        config = {
            "model_type": args["model_type"],
            "feature_method": args["feature_method"],
            "window_size": args["window_size"],
            "stride": args["stride"],
            "epochs": args["epochs"],
            "batch_size": args["batch_size"],
            "lr": args["lr"],
            "seed": args["seed"],
            "use_val": args["use_val"],
            "use_class_weight": args["use_class_weight"],
            "data_folder": args["data_folder"],
            "topk": args.get("topk", 300),
            "stat_window": args.get("stat_window", 5),
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
    if args["test_noise"] > 0:
        Xte_m = Xte_m + args["test_noise"] * Xte_m * np.random.randn(*Xte_m.shape)

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
