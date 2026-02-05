import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_class_weight(y):
    """
    역빈도 기반 클래스 가중치 계산.
    weight_c = N_total / (n_classes * N_c)
    Returns dict {class_id: weight} for keras model.fit(class_weight=...)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = n_samples / (n_classes * counts)
    return {int(c): float(w) for c, w in zip(classes, weights)}


class SplitWithVal:
    def __init__(self, group_size=10, val_ratio=0.1, test_ratio=0.2, seed=0):
        self.group_size = int(group_size)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.seed = int(seed)

    def split(self, X, y):
        rng = np.random.default_rng(self.seed)

        N, D = X.shape
        G = self.group_size
        n_groups = N // G
        if n_groups <= 1:
            raise ValueError(f"그룹 수가 너무 적습니다. N={N}, G={G}")

        Xg = X[: n_groups * G].reshape(n_groups, G, D)
        yg = y[: n_groups * G].reshape(n_groups, G)

        group_labels = yg[:, 0]  # 그룹 대표 라벨
        train_g, val_g, test_g = [], [], []

        for label in np.unique(group_labels):
            idx = np.where(group_labels == label)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_test = int(n * self.test_ratio)
            n_val = int(n * self.val_ratio)

            test_g += idx[:n_test].tolist()
            val_g += idx[n_test : n_test + n_val].tolist()
            train_g += idx[n_test + n_val :].tolist()

        rng.shuffle(train_g)
        rng.shuffle(val_g)
        # rng.shuffle(test_g)

        Xtr = Xg[train_g].reshape(-1, D)
        ytr = yg[train_g].reshape(-1)

        Xva = Xg[val_g].reshape(-1, D)
        yva = yg[val_g].reshape(-1)

        Xte = Xg[test_g].reshape(-1, D)
        yte = yg[test_g].reshape(-1)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xva = scaler.transform(Xva)
        Xte = scaler.transform(Xte)

        return Xtr, ytr, Xva, yva, Xte, yte


class SplitWithoutVal:
    def __init__(self, group_size=10, test_ratio=0.2, seed=0):
        self.group_size = int(group_size)
        self.test_ratio = float(test_ratio)
        self.seed = int(seed)

    def split(self, X, y):
        rng = np.random.default_rng(self.seed)

        N, D = X.shape
        G = self.group_size
        n_groups = N // G
        if n_groups <= 1:
            raise ValueError(f"그룹 수가 너무 적습니다. N={N}, G={G}")

        Xg = X[: n_groups * G].reshape(n_groups, G, D)
        yg = y[: n_groups * G].reshape(n_groups, G)

        group_labels = yg[:, 0]
        train_g, test_g = [], []

        for label in np.unique(group_labels):
            idx = np.where(group_labels == label)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_test = int(n * self.test_ratio)
            test_g += idx[:n_test].tolist()
            train_g += idx[n_test:].tolist()

        rng.shuffle(train_g)

        Xtr = Xg[train_g].reshape(-1, D)
        ytr = yg[train_g].reshape(-1)

        Xte = Xg[test_g].reshape(-1, D)
        yte = yg[test_g].reshape(-1)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        return Xtr, ytr, Xte, yte