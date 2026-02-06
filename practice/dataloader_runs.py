"""
런 단위 데이터 로더 (GPT 제안 3번)

파일 경계를 보존하여 그룹/윈도우가 서로 다른 파일의 데이터를 섞지 않도록 함
"""
import os
import re
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from multiprocessing import Pool

from .dataloader import LABELS, LABEL2ID, ID2LABEL, infer_label_id, extract_number


def load_Xy_runs(folder_path: str, include_time: bool = False, n_workers=None, verbose=True):
    """
    런 단위로 데이터 로드 (GPT 제안 3번)

    Returns:
        X_runs: List[np.ndarray] - 각 파일(런)별 데이터
        y_runs: List[np.ndarray] - 각 파일(런)별 라벨
        feature_names: list - 피처 이름
    """
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    file_list = [f for f in file_list if infer_label_id(f) is not None]

    def sort_key(f):
        return (infer_label_id(f), extract_number(f), f)

    file_list.sort(key=sort_key)

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    def load_one_csv_run(file_name):
        """런 단위로 로드 (파일 = 런)"""
        label_id = infer_label_id(file_name)
        if label_id is None:
            return None

        fp = os.path.join(folder_path, file_name)
        df = pd.read_csv(fp)

        if (not include_time) and ("KCNTOMS" in df.columns):
            df = df.drop(columns=["KCNTOMS"])

        X_run = df.to_numpy(dtype=np.float32)
        y_run = np.full((len(df),), label_id, dtype=np.int64)

        return X_run, y_run, file_name

    func_fixed = partial(lambda f: load_one_csv_run(f))

    with Pool(processes=n_workers) as pool:
        results = list(pool.imap(func_fixed, file_list))

    results = [r for r in results if r is not None]
    if len(results) == 0:
        raise RuntimeError("라벨과 매칭되는 CSV가 없습니다.")

    # 첫 파일에서 feature_names 추출
    feature_names = list(results[0][0][0].columns) if isinstance(results[0][0], pd.DataFrame) else \
                    [f"feature_{i}" for i in range(results[0][0].shape[1])]

    # 런 단위로 분리
    X_runs = [X for X, y, fname in results]
    y_runs = [y for X, y, fname in results]
    file_names = [fname for X, y, fname in results]

    if verbose:
        print("\n[Data Class Distribution - Run-based]")
        print(f"Total runs: {len(X_runs)}")
        total_samples = sum(len(X) for X in X_runs)
        print(f"Total samples: {total_samples}")

        # 클래스별 통계
        label_counts = {}
        for y in y_runs:
            label = int(y[0])
            label_counts[label] = label_counts.get(label, 0) + len(y)

        for label_id in sorted(label_counts.keys()):
            class_name = ID2LABEL.get(label_id, f"UNKNOWN({label_id})")
            count = label_counts[label_id]
            print(f"  {class_name:15s}: {count:6d} samples")

    # 쓸모없는 변수 제거 (기본값)
    from .dataloader import load_Xy
    # 원래 load_Xy의 exclude_useless_features 로직 재사용
    # 간단히 하기 위해 여기서는 스킵하고 나중에 전처리에서 처리

    return X_runs, y_runs, feature_names


def create_sliding_windows_runs(X_runs, y_runs, window_size):
    """
    런 단위로 슬라이딩 윈도우 생성 (파일 경계 보존)

    Args:
        X_runs: List[np.ndarray] - 각 (N_i, D)
        y_runs: List[np.ndarray] - 각 (N_i,)
        window_size: int

    Returns:
        X_windows: (총 윈도우 수, window_size, D)
        y_windows: (총 윈도우 수,) - 마지막 타임스텝 라벨
    """
    all_X_windows = []
    all_y_windows = []

    for X_run, y_run in zip(X_runs, y_runs):
        N, D = X_run.shape

        # 이 런에서 생성 가능한 윈도우 개수
        num_windows = N - window_size + 1

        if num_windows <= 0:
            continue

        for i in range(num_windows):
            X_window = X_run[i : i + window_size]  # (window_size, D)
            y_window = y_run[i + window_size - 1]  # 마지막 타임스텝 라벨

            all_X_windows.append(X_window)
            all_y_windows.append(y_window)

    return np.array(all_X_windows), np.array(all_y_windows)
