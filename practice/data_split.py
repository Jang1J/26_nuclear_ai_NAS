"""
런(파일) 단위 데이터 분할.

핵심: 같은 런(파일)의 데이터가 train과 test에 동시에 들어가지 않도록 보장.
-> 데이터 누수(leakage) 방지.
"""
import numpy as np


def compute_class_weight(y):
    """
    역빈도 기반 클래스 가중치 계산.
    weight_c = N_total / (n_classes * N_c)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = n_samples / (n_classes * counts)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def split_runs(X_runs, y_runs, run_names, val_ratio=0.1, test_ratio=0.2, seed=0, use_val=True):
    """
    런(파일) 단위로 train/val/test 분할. 같은 런이 여러 세트에 걸치지 않음.

    Args:
        X_runs: List[np.ndarray] - 각 런의 (N_i, D)
        y_runs: List[np.ndarray] - 각 런의 (N_i,)
        run_names: List[str] - 파일명
        val_ratio: float
        test_ratio: float
        seed: int
        use_val: bool

    Returns:
        train_runs, val_runs, test_runs: 각각 (X_list, y_list) 튜플
        val_runs는 use_val=False이면 ([], [])
    """
    rng = np.random.default_rng(seed)

    # 런별 라벨 (각 런은 단일 클래스)
    run_labels = np.array([int(y[0]) for y in y_runs])

    train_idx, val_idx, test_idx = [], [], []

    for label in np.unique(run_labels):
        idx = np.where(run_labels == label)[0]
        rng.shuffle(idx)
        n = len(idx)

        if n == 1:
            # 런이 1개뿐이면 train에만 배정 (test 불가)
            train_idx.extend(idx)
            continue

        if n == 2:
            # 런이 2개면 train 1, test 1 (val 없음)
            train_idx.append(idx[0])
            test_idx.append(idx[1])
            continue

        # 런이 3개 이상
        n_test = max(1, int(n * test_ratio))

        if use_val:
            # train에 최소 n//2 개 보장하면서 val 배정
            n_val = max(1, int(n * val_ratio)) if n >= 5 else 0
            # train에 최소 1개 보장
            while n_test + n_val >= n and n_val > 0:
                n_val -= 1
            if n_test + n_val >= n:
                n_test = n - 1
                n_val = 0
        else:
            n_val = 0

        # train에 최소 1개 보장
        if n_test + n_val >= n:
            n_test = max(1, n - 1)
            n_val = 0

        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test : n_test + n_val])
        train_idx.extend(idx[n_test + n_val :])

    rng.shuffle(train_idx)

    train_X = [X_runs[i] for i in train_idx]
    train_y = [y_runs[i] for i in train_idx]

    test_X = [X_runs[i] for i in test_idx]
    test_y = [y_runs[i] for i in test_idx]

    if use_val:
        val_X = [X_runs[i] for i in val_idx]
        val_y = [y_runs[i] for i in val_idx]
    else:
        val_X, val_y = [], []

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)
