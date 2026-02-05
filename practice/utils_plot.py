import os
import numpy as np


def save_acc_loss(history, acc_path, loss_path):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(acc_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(loss_path) or ".", exist_ok=True)

    hist = history.history
    plt.figure()
    plt.plot(hist.get("accuracy", []), label="train_acc")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(hist.get("loss", []), label="train_loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()


def save_confusion_matrix(y_true, y_pred, labels, cm_path, available_classes=None):
    """
    혼동 행렬 저장.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        labels: 클래스 이름 리스트
        cm_path: 저장 경로
        available_classes: 실제 학습에 사용된 클래스 ID (None이면 labels 기준)
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    os.makedirs(os.path.dirname(cm_path) or ".", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    # 누락된 클래스 표시 (회색 배경)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')

    # 누락된 클래스는 회색으로 표시
    if available_classes is not None:
        missing_indices = [i for i in range(len(labels)) if i not in available_classes]
        for idx in missing_indices:
            # 행과 열에 회색 배경
            ax.add_patch(plt.Rectangle((idx - 0.5, -0.5), 1, len(labels),
                                       fill=True, color='gray', alpha=0.3, zorder=0))
            ax.add_patch(plt.Rectangle((-0.5, idx - 0.5), len(labels), 1,
                                       fill=True, color='gray', alpha=0.3, zorder=0))

    ax.set_title("Confusion Matrix" + (" (Gray: Missing Classes)" if available_classes is not None else ""))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # 숫자 표시
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                   fontsize=8, color=color)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()


def save_per_class_accuracy(y_true, y_pred, labels, save_path, available_classes=None):
    """
    클래스별 정확도(recall) 바 차트.
    어떤 클래스에서 성능이 부족한지 한눈에 파악 가능.

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        labels: 클래스 이름 리스트
        save_path: 저장 경로
        available_classes: 실제 학습에 사용된 클래스 ID (None이면 labels 기준)
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    n_classes = len(labels)
    accs = []
    counts = []
    colors = []

    for i in range(n_classes):
        mask = (y_true == i)
        counts.append(int(mask.sum()))
        if mask.sum() > 0:
            accs.append(float((y_pred[mask] == i).mean()))
            colors.append("steelblue")
        else:
            accs.append(0.0)
            # 누락된 클래스는 회색
            if available_classes is not None and i not in available_classes:
                colors.append("gray")
            else:
                colors.append("lightgray")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(n_classes)

    bars = ax1.bar(x, accs, color=colors, alpha=0.8)
    ax1.set_ylabel("Per-class Accuracy (Recall)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylim(0, 1.15)

    for bar, acc, count, color in zip(bars, accs, counts, colors):
        text = f"{acc:.2f}\n(n={count})" if count > 0 else "N/A"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            text,
            ha="center", va="bottom", fontsize=9,
        )

    title = "Per-class Accuracy"
    if available_classes is not None:
        title += " (Gray: Missing Classes)"
    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()