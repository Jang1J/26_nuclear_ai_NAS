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


def save_confusion_matrix(y_true, y_pred, labels, cm_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    os.makedirs(os.path.dirname(cm_path) or ".", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()


def save_per_class_accuracy(y_true, y_pred, labels, save_path):
    """
    클래스별 정확도(recall) 바 차트.
    어떤 클래스에서 성능이 부족한지 한눈에 파악 가능.
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    n_classes = len(labels)
    accs = []
    counts = []
    for i in range(n_classes):
        mask = (y_true == i)
        counts.append(int(mask.sum()))
        if mask.sum() > 0:
            accs.append(float((y_pred[mask] == i).mean()))
        else:
            accs.append(0.0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(n_classes)

    bars = ax1.bar(x, accs, color="steelblue", alpha=0.8)
    ax1.set_ylabel("Per-class Accuracy (Recall)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylim(0, 1.15)

    for bar, acc, count in zip(bars, accs, counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{acc:.2f}\n(n={count})",
            ha="center", va="bottom", fontsize=9,
        )

    ax1.set_title("Per-class Accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()