import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from sklearn.metrics import classification_report

from .dataloader import load_Xy, create_sliding_windows_grouped, LABELS, ID2LABEL
from .feature_method import make_feature_method
from .data_split import SplitWithVal, SplitWithoutVal, compute_class_weight
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
        print(f"  {ID2LABEL.get(int(l), f'UNKNOWN({l})'):10s}: {c:6d} ({c/total:.3f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", type=str, required=True)

    p.add_argument("--model_folder", type=str, default="models")
    p.add_argument("--train_folder", type=str, default="train_results")
    p.add_argument("--test_folder", type=str, default="test_results")

    # 모델 선택 (5가지)
    p.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["mlp", "cnn", "cnn_attention", "lstm", "transformer"],
        help="모델 종류: mlp, cnn, cnn_attention, lstm, transformer"
    )

    # 피처 선택
    p.add_argument(
        "--feature_method",
        type=str,
        default="all",
        choices=["all", "change", "selection", "diff", "stats", "physics"],
    )

    p.add_argument("--group_size", type=int, default=10)
    p.add_argument("--use_val", action="store_true")

    # 학습 설정
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (기본 1e-3)")

    p.add_argument("--train", action="store_true")
    p.add_argument("--test_noise", type=float, default=0.0)

    # 추가 옵션
    p.add_argument("--use_class_weight", action="store_true", help="역빈도 기반 클래스 가중치 적용")
    p.add_argument("--window_size", type=int, default=10, help="시퀀스 모델용 슬라이딩 윈도우 크기")

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
        return make_feature_method("physics", group_size=args["group_size"], moving_std_window=3)
    return make_feature_method(args["feature_method"])


def _build_model_single(args, D, W=None):
    """5가지 모델 중 하나를 빌드"""
    model_type = args["model_type"]

    if model_type == "mlp":
        # MLP는 시계열이 아닌 마지막 타임스텝만 사용
        return build_mlp(D, len(LABELS))

    # 나머지 4개 모델은 모두 시계열 입력 필요
    if W is None:
        raise ValueError(f"{model_type} requires window_size")

    if model_type == "cnn":
        return build_cnn(W, D, len(LABELS))
    if model_type == "cnn_attention":
        return build_cnn_attention(W, D, len(LABELS))
    if model_type == "lstm":
        return build_lstm(W, D, len(LABELS))
    if model_type == "transformer":
        return build_transformer(W, D, len(LABELS))

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
            monitor=monitor_metric,
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        )
    )
    if args["use_val"] and Xva is not None:
        cb.append(
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
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


def _predict_proba(model, X):
    t0 = time.time()
    y_prob = model.predict(X, verbose=0)
    infer_time = time.time() - t0
    return y_prob, infer_time


def run_single(args):
    # 시퀀스 모델 여부
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
        run_name += f"__win={args['window_size']}"

    model_dir = Path(args["model_folder"])
    train_dir = Path(args["train_folder"]) / run_name
    test_dir = Path(args["test_folder"]) / run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{run_name}__model.keras"
    train_acc_path = train_dir / "acc_vs_epoch.png"
    train_loss_path = train_dir / "loss_vs_epoch.png"
    cm_path = test_dir / "confusion_matrix.png"
    per_class_path = test_dir / "per_class_accuracy.png"
    metrics_path = test_dir / "test_metrics.txt"

    # 1) 데이터 로드
    X, y, feature_names = load_Xy(args["data_folder"], include_time=False)
    print("[Data]")
    print("X:", X.shape, "y:", y.shape)
    print("features:", len(feature_names))

    # 2) 피처
    feat = _build_feature_engineer(args, model_dir, train_dir)
    X2, y2, feats2 = feat.fit_transform(X, y, feature_names)
    print(f"\n[Feature method] {args['feature_method']}")
    print("X:", X.shape, "->", X2.shape)
    print("features:", len(feature_names), "->", len(feats2))

    # 3) 데이터 분할
    if args["use_val"]:
        splitter = SplitWithVal(group_size=args["group_size"], val_ratio=0.1, test_ratio=0.2, seed=args["seed"])
        Xtr, ytr, Xva, yva, Xte, yte = splitter.split(X2, y2)
        print_label_dist("\ntrain", ytr)
        print_label_dist("val", yva)
        print_label_dist("test", yte)
    else:
        splitter = SplitWithoutVal(group_size=args["group_size"], test_ratio=0.2, seed=args["seed"])
        Xtr, ytr, Xte, yte = splitter.split(X2, y2)
        Xva, yva = None, None
        print_label_dist("\ntrain", ytr)
        print_label_dist("test", yte)

    # 4) class_weight
    class_weight = None
    if args["use_class_weight"]:
        class_weight = compute_class_weight(ytr)
        print("\n[Class weights]")
        for cid, w in sorted(class_weight.items()):
            print(f"  {ID2LABEL.get(cid, cid):10s}: {w:.3f}")

    # 5) 모델 선택 + 데이터 reshape
    D = Xtr.shape[1]

    if is_seq_model:
        W = args["window_size"]
        G = args["group_size"]

        Xtr_m, ytr = create_sliding_windows_grouped(Xtr, ytr, W, G)
        Xte_m, yte = create_sliding_windows_grouped(Xte, yte, W, G)
        Xva_m = None
        if args["use_val"]:
            Xva_m, yva = create_sliding_windows_grouped(Xva, yva, W, G)

        print(f"\n[Window] size={W}, train: {Xtr_m.shape}, test: {Xte_m.shape}")
        model = _build_model_single(args, D=D, W=W)

    else:
        Xtr_m, Xte_m = Xtr, Xte
        Xva_m = Xva if args["use_val"] else None
        model = _build_model_single(args, D=D)

    print(f"\n[Model] {args['model_type']}")
    model.summary()

    # 6) 학습 or 로드
    history = None
    train_time = 0.0

    if args["train"]:
        print("\n[Mode] TRAIN")
        _compile_model(model, args)
        history, train_time = _fit_model(model, Xtr_m, ytr, Xva_m, yva, args, class_weight=class_weight)
        print(f"\n[Training time] {train_time:.1f}s")

        model.save(model_path)
        print("[Saved model]", model_path)

        # 학습에 사용된 클래스 정보 저장
        available_classes = np.unique(ytr)
        class_mapping_path = model_dir / f"{run_name}__class_mapping.npy"
        np.save(class_mapping_path, available_classes)
        print(f"[Saved class mapping] {class_mapping_path}")
        print(f"  Available classes: {[ID2LABEL[int(c)] for c in available_classes]}")

        if history is not None:
            save_acc_loss(history, str(train_acc_path), str(train_loss_path))
            print("[Saved train curves]", train_acc_path, train_loss_path)

    else:
        print("\n[Mode] LOAD")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("[Loaded model]", model_path)

        # 클래스 매핑 정보 로드 (있으면)
        class_mapping_path = model_dir / f"{run_name}__class_mapping.npy"
        if class_mapping_path.exists():
            available_classes = np.load(class_mapping_path)
            print(f"[Loaded class mapping] {class_mapping_path}")
            print(f"  Available classes: {[ID2LABEL[int(c)] for c in available_classes]}")
        else:
            print("[Warning] No class mapping file found. Using all classes.")

    # 7) 테스트
    if args["test_noise"] > 0:
        Xte_m = Xte_m + args["test_noise"] * Xte_m * np.random.randn(*Xte_m.shape)

    test_loss, test_acc = model.evaluate(Xte_m, yte, verbose=0)

    y_prob, inference_time = _predict_proba(model, Xte_m)
    y_pred = np.argmax(y_prob, axis=1)

    n_test = len(yte)
    per_sample_ms = (inference_time / n_test) * 1000

    # 실제 데이터에 존재하는 클래스만 사용
    unique_classes = np.unique(yte)
    target_names = [LABELS[i] for i in unique_classes]

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_acc : {test_acc:.6f}\n")
        f.write(f"inference_time: {inference_time:.3f}s ({per_sample_ms:.4f}ms/sample)\n")
        if args["train"]:
            f.write(f"train_time: {train_time:.1f}s\n")
        f.write(f"\n{classification_report(yte, y_pred, labels=unique_classes, target_names=target_names)}\n")

    print("\n[Saved test metrics]", metrics_path)
    print("\n[Test result]")
    print(f"  loss: {test_loss:.6f}")
    print(f"  acc : {test_acc:.6f}")
    print(f"  inference: {inference_time:.3f}s total, {per_sample_ms:.4f}ms/sample")

    # 8) 시각화
    # 9개 클래스 전체 구조로 시각화 (누락된 클래스는 회색 표시)
    available_classes_viz = unique_classes if args["train"] else None

    # 전체 9개 클래스 기준으로 시각화
    all_labels = LABELS
    save_confusion_matrix(yte, y_pred, all_labels, str(cm_path), available_classes=unique_classes)
    print("[Saved confusion matrix]", cm_path)

    save_per_class_accuracy(yte, y_pred, all_labels, str(per_class_path), available_classes=unique_classes)
    print("[Saved per-class accuracy]", per_class_path)


def main():
    args = vars(parse_args())
    np.random.seed(args["seed"])
    tf.random.set_seed(args["seed"])

    run_single(args)


if __name__ == "__main__":
    main()
