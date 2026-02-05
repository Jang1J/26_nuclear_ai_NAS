import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from .dataloader import load_Xy, create_sliding_windows_grouped, LABELS, ID2LABEL
from .feature_method import make_feature_method
from .data_split import SplitWithVal, SplitWithoutVal, compute_class_weight
from .model import (
    build_mlp,
    build_cnn_2d,
    to_square_per_timestep,
    build_mlp_v2,
    build_cnn1d,
    build_lstm,
    build_cnn1d_lstm,
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

    # 단일 모델 모드
    p.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "mlp_v2", "cnn1d", "lstm", "hybrid"],
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

    # 앙상블 모드
    p.add_argument("--ensemble", action="store_true", help="3모델 앙상블 모드(MLP_v2 + CNN1D + Hybrid)")
    p.add_argument(
        "--ensemble_method",
        type=str,
        default="soft_vote",
        choices=["soft_vote", "weighted_vote", "stacking"],
        help="앙상블 결합 방식",
    )

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
    model_type = args["model_type"]
    if model_type == "mlp":
        return build_mlp(D, len(LABELS))
    if model_type == "mlp_v2":
        return build_mlp_v2(D, len(LABELS))
    if model_type == "cnn":
        # build_cnn_2d는 입력 shape 필요. 호출 전에 to_square_per_timestep로 S를 얻어야 함.
        raise RuntimeError("cnn(2D)는 _build_model_single에서 직접 생성하지 않습니다.")
    if model_type == "cnn1d":
        if W is None:
            raise ValueError("cnn1d requires window_size")
        return build_cnn1d(W, D, len(LABELS))
    if model_type == "lstm":
        if W is None:
            raise ValueError("lstm requires window_size")
        return build_lstm(W, D, len(LABELS))
    if model_type == "hybrid":
        if W is None:
            raise ValueError("hybrid requires window_size")
        return build_cnn1d_lstm(W, D, len(LABELS))
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
    is_seq_model = args["model_type"] in ("cnn1d", "lstm", "hybrid")

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

    if args["model_type"] == "cnn":
        Xtr_m, S = to_square_per_timestep(Xtr)
        Xte_m, _ = to_square_per_timestep(Xte)
        Xva_m = None
        if args["use_val"]:
            Xva_m, _ = to_square_per_timestep(Xva)
        model = build_cnn_2d((S, S, 1), len(LABELS))

    elif is_seq_model:
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


def run_ensemble(args):
    if not args.get("use_val", False):
        raise ValueError("Ensemble mode requires --use_val (val set is needed for weights/stacking).")

    W = args["window_size"]
    G = args["group_size"]

    run_name = (
        f"ensemble__method={args['ensemble_method']}"
        f"__feat={args['feature_method']}"
        f"__val=1"
        f"__ep={args['epochs']}"
        f"__cw={int(args['use_class_weight'])}"
        f"__seed={args['seed']}"
        f"__win={W}"
    )

    model_dir = Path(args["model_folder"])
    train_dir = Path(args["train_folder"]) / run_name
    test_dir = Path(args["test_folder"]) / run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

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

    # 2) 피처 엔지니어링
    feat = _build_feature_engineer(args, model_dir, train_dir)
    X2, y2, feats2 = feat.fit_transform(X, y, feature_names)
    print(f"\n[Feature method] {args['feature_method']}")
    print("X:", X.shape, "->", X2.shape)
    print("features:", len(feature_names), "->", len(feats2))

    # 3) 데이터 분할
    splitter = SplitWithVal(group_size=G, val_ratio=0.1, test_ratio=0.2, seed=args["seed"])
    Xtr, ytr, Xva, yva, Xte, yte = splitter.split(X2, y2)
    print_label_dist("\ntrain", ytr)
    print_label_dist("val", yva)
    print_label_dist("test", yte)

    # 4) class_weight (flat 기준)
    class_weight = None
    if args["use_class_weight"]:
        class_weight = compute_class_weight(ytr)
        print("\n[Class weights]")
        for cid, w in sorted(class_weight.items()):
            print(f"  {ID2LABEL.get(cid, cid):10s}: {w:.3f}")

    # 5) 슬라이딩 윈도우 (3개 모델 입력 정렬의 핵심)
    D = Xtr.shape[1]
    Xtr_win, ytr_win = create_sliding_windows_grouped(Xtr, ytr, W, G)
    Xva_win, yva_win = create_sliding_windows_grouped(Xva, yva, W, G)
    Xte_win, yte_win = create_sliding_windows_grouped(Xte, yte, W, G)

    print(f"\n[Window] size={W}")
    print("  train:", Xtr_win.shape, "val:", Xva_win.shape, "test:", Xte_win.shape)

    # MLP는 마지막 timestep만 사용 (샘플 수 정렬 유지)
    Xtr_mlp = Xtr_win[:, -1, :]
    Xva_mlp = Xva_win[:, -1, :]
    Xte_mlp = Xte_win[:, -1, :]

    # 6) 3개 모델 순차 학습
    histories = {}
    train_times = {}
    val_accs = {}
    val_probs = {}
    test_probs = {}

    def train_one(tag, model, Xtr_in, ytr_in, Xva_in, yva_in, Xte_in):
        print(f"\n[Train] {tag}")
        model.summary()

        if args["train"]:
            _compile_model(model, args)
            hist, t = _fit_model(model, Xtr_in, ytr_in, Xva_in, yva_in, args, class_weight=class_weight)
            histories[tag] = hist
            train_times[tag] = t
            model_path = model_dir / f"{run_name}__{tag}__model.keras"
            model.save(model_path)
            print("[Saved model]", model_path)

            # 클래스 매핑 저장 (앙상블의 경우 첫 번째 모델에만)
            if tag == "mlp_v2":
                available_classes = np.unique(ytr_in)
                class_mapping_path = model_dir / f"{run_name}__class_mapping.npy"
                np.save(class_mapping_path, available_classes)
                print(f"[Saved class mapping] {class_mapping_path}")
        else:
            model_path = model_dir / f"{run_name}__{tag}__model.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            model = tf.keras.models.load_model(model_path)
            print("[Loaded model]", model_path)

            # 클래스 매핑 로드 (앙상블의 경우 첫 번째 모델에만)
            if tag == "mlp_v2":
                class_mapping_path = model_dir / f"{run_name}__class_mapping.npy"
                if class_mapping_path.exists():
                    available_classes = np.load(class_mapping_path)
                    print(f"[Loaded class mapping] Available: {[ID2LABEL[int(c)] for c in available_classes]}")

        # val / test 확률
        v_prob, _ = _predict_proba(model, Xva_in)
        t_prob, _ = _predict_proba(model, Xte_in)

        v_pred = np.argmax(v_prob, axis=1)
        val_acc = float(np.mean(v_pred == yva_in))
        print(f"  [val_acc] {val_acc:.6f}")

        val_accs[tag] = val_acc
        val_probs[tag] = v_prob
        test_probs[tag] = t_prob

        return model

    # MLP_v2
    _ = train_one("mlp_v2", build_mlp_v2(D, len(LABELS)), Xtr_mlp, ytr_win, Xva_mlp, yva_win, Xte_mlp)
    # CNN1D
    _ = train_one("cnn1d", build_cnn1d(W, D, len(LABELS)), Xtr_win, ytr_win, Xva_win, yva_win, Xte_win)
    # Hybrid
    _ = train_one("hybrid", build_cnn1d_lstm(W, D, len(LABELS)), Xtr_win, ytr_win, Xva_win, yva_win, Xte_win)

    # train curve 저장: 마지막 학습한 history 기준으로 저장 (원하면 확장 가능)
    if args["train"] and histories:
        # 가장 최근 history
        last_key = list(histories.keys())[-1]
        save_acc_loss(histories[last_key], str(train_acc_path), str(train_loss_path))
        print("[Saved train curves]", train_acc_path, train_loss_path)

    # 7) 앙상블 결합
    probs_val = [val_probs[k] for k in ["mlp_v2", "cnn1d", "hybrid"]]
    probs_test = [test_probs[k] for k in ["mlp_v2", "cnn1d", "hybrid"]]

    if args["ensemble_method"] == "soft_vote":
        y_prob_ens = np.mean(probs_test, axis=0)
        weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

    elif args["ensemble_method"] == "weighted_vote":
        w = np.array([val_accs["mlp_v2"], val_accs["cnn1d"], val_accs["hybrid"]], dtype=np.float64)
        w = w / (w.sum() + 1e-12)
        weights = w
        y_prob_ens = weights[0] * probs_test[0] + weights[1] * probs_test[1] + weights[2] * probs_test[2]

    elif args["ensemble_method"] == "stacking":
        # meta learner: val set 예측 확률을 concat
        X_meta_val = np.concatenate(probs_val, axis=1)
        X_meta_test = np.concatenate(probs_test, axis=1)

        meta = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="auto")
        meta.fit(X_meta_val, yva_win)
        y_prob_ens = meta.predict_proba(X_meta_test)
        weights = None

    else:
        raise ValueError(f"Unknown ensemble_method: {args['ensemble_method']}")

    y_pred_ens = np.argmax(y_prob_ens, axis=1)
    test_acc = float(np.mean(y_pred_ens == yte_win))

    # 8) 저장
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"[run_name] {run_name}\n")
        f.write(f"[ensemble_method] {args['ensemble_method']}\n")
        f.write(f"[window_size] {W}\n")
        f.write("\n[val_acc_each]\n")
        for k in ["mlp_v2", "cnn1d", "hybrid"]:
            f.write(f"  {k}: {val_accs[k]:.6f}\n")
        if weights is not None:
            f.write("\n[weights]\n")
            f.write(f"  mlp_v2: {weights[0]:.6f}\n")
            f.write(f"  cnn1d : {weights[1]:.6f}\n")
            f.write(f"  hybrid: {weights[2]:.6f}\n")
        f.write(f"\n[test_acc_ensemble]: {test_acc:.6f}\n")
        if args["train"]:
            f.write("\n[train_time]\n")
            for k in ["mlp_v2", "cnn1d", "hybrid"]:
                f.write(f"  {k}: {train_times.get(k, 0.0):.1f}s\n")

        # 실제 데이터에 존재하는 클래스만 사용
        unique_classes = np.unique(yte_win)
        target_names = [LABELS[i] for i in unique_classes]
        f.write(f"\n{classification_report(yte_win, y_pred_ens, labels=unique_classes, target_names=target_names)}\n")

    print("\n[Saved test metrics]", metrics_path)
    print("\n[Test result] ensemble")
    print(f"  acc : {test_acc:.6f}")

    # 시각화 (9개 클래스 전체 구조)
    all_labels = LABELS
    save_confusion_matrix(yte_win, y_pred_ens, all_labels, str(cm_path), available_classes=unique_classes)
    print("[Saved confusion matrix]", cm_path)

    save_per_class_accuracy(yte_win, y_pred_ens, all_labels, str(per_class_path), available_classes=unique_classes)
    print("[Saved per-class accuracy]", per_class_path)


def main():
    args = vars(parse_args())
    np.random.seed(args["seed"])
    tf.random.set_seed(args["seed"])

    if args.get("ensemble"):
        run_ensemble(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
