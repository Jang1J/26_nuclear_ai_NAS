import os
import re
import json
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from pathlib import Path
from multiprocessing import Pool

# 새 라벨링 체계 (9개 클래스)
LABELS = [
    "NORMAL",       # 0: 정상
    "LOCA_HL",      # 1: Hot Leg LOCA
    "LOCA_CL",      # 2: Cold Leg LOCA
    "LOCA_RCP",     # 3: RCP Seal LOCA
    "SGTR_Loop1",   # 4: SGTR 루프 1
    "SGTR_Loop2",   # 5: SGTR 루프 2
    "SGTR_Loop3",   # 6: SGTR 루프 3
    "ESDE_in",      # 7: ESDE Inside
    "ESDE_out",     # 8: ESDE Outside
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


def infer_label_id(file_name: str):
    """
    파일명 prefix로 라벨 id 추론.
    라벨링:
      NORMAL.csv / NORMAL_*.csv  -> NORMAL (0)
      LOCA_HL_*.csv              -> LOCA_HL (1)
      LOCA_CL_*.csv              -> LOCA_CL (2)
      LOCA_RCP_*.csv             -> LOCA_RCP (3)
      SGTR_Loop1_*.csv           -> SGTR_Loop1 (4)
      SGTR_Loop2_*.csv           -> SGTR_Loop2 (5)
      SGTR_Loop3_*.csv           -> SGTR_Loop3 (6)
      ESDE_in_Loop*_*.csv        -> ESDE_in (7)
      ESDE_out_Loop*_*.csv       -> ESDE_out (8)
    """
    base = os.path.basename(file_name)

    # NORMAL (NORMAL.csv 또는 NORMAL_*.csv)
    if base == "NORMAL.csv" or base.startswith("NORMAL_"):
        return LABEL2ID["NORMAL"]

    # LOCA 세분화 (순서 중요: HL, CL 먼저 매칭 후 RCP)
    if base.startswith("LOCA_HL_"):
        return LABEL2ID["LOCA_HL"]
    if base.startswith("LOCA_CL_"):
        return LABEL2ID["LOCA_CL"]
    if base.startswith("LOCA_RCP_"):
        return LABEL2ID["LOCA_RCP"]

    # SGTR 세분화 (Loop1, Loop2, Loop3)
    if base.startswith("SGTR_Loop1_"):
        return LABEL2ID["SGTR_Loop1"]
    if base.startswith("SGTR_Loop2_"):
        return LABEL2ID["SGTR_Loop2"]
    if base.startswith("SGTR_Loop3_"):
        return LABEL2ID["SGTR_Loop3"]

    # ESDE (Inside/Outside) — Loop 번호는 무시하고 in/out만 구분
    # ESDE_out_Loop3__leak 같은 오타(언더스코어 2개)도 매칭
    if base.startswith("ESDE_in_"):
        return LABEL2ID["ESDE_in"]
    if base.startswith("ESDE_out_"):
        return LABEL2ID["ESDE_out"]

    return None


def extract_number(fname: str) -> int:
    """파일명 끝의 번호를 정렬키로 사용."""
    base = os.path.basename(fname)
    nums = re.findall(r"(\d+)", base)
    return int(nums[-1]) if nums else -1


def _resolve_useless_json_path(useless_features_json: str) -> str:
    """
    useless_features json 파일을 CWD에 무관하게 찾는다.

    탐색 우선순위:
      1) 입력 경로가 이미 실제 파일이면 그대로 사용
      2) 프로젝트 루트(practice/ 상위 폴더 = NAS/) 기준
      3) CWD 기준 (기존 동작 호환)
    """
    p = Path(useless_features_json)
    if p.is_file():
        return str(p)

    # practice/dataloader.py 기준 → 부모의 부모 = NAS/
    project_root = Path(__file__).resolve().parents[1]
    cand = project_root / useless_features_json
    if cand.is_file():
        return str(cand)

    cand = Path.cwd() / useless_features_json
    if cand.is_file():
        return str(cand)

    return useless_features_json  # 못 찾으면 원래대로 (이후 exists 체크에서 skip)


def load_one_csv(file_name: str, folder_path: str, include_time: bool = False):
    label_id = infer_label_id(file_name)
    if label_id is None:
        return None

    fp = os.path.join(folder_path, file_name)
    df = pd.read_csv(fp)

    if (not include_time) and ("KCNTOMS" in df.columns):
        df = df.drop(columns=["KCNTOMS"])

    y = np.full((len(df),), label_id, dtype=np.int64)
    return df, y


def load_Xy_runs(folder_path: str, include_time: bool = False, n_workers=None, verbose=True,
                 exclude_useless_features: bool = True, useless_features_json: str = "useless_features_all.json"):
    """
    런(파일) 단위로 데이터 로드. 파일 경계를 보존.

    Returns:
        X_runs: List[np.ndarray] - 각 파일(런)별 데이터
        y_runs: List[np.ndarray] - 각 파일(런)별 라벨
        run_names: List[str] - 파일명
        feature_names: list
    """
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    file_list = [f for f in file_list if infer_label_id(f) is not None]

    def sort_key(f):
        return (infer_label_id(f), extract_number(f), f)

    file_list.sort(key=sort_key)

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    func_fixed = partial(load_one_csv, folder_path=folder_path, include_time=include_time)

    with Pool(processes=n_workers) as pool:
        results = list(pool.imap(func_fixed, file_list))

    valid = [(r, fname) for r, fname in zip(results, file_list) if r is not None]
    if len(valid) == 0:
        raise RuntimeError("라벨과 매칭되는 CSV가 없습니다.")

    feature_names = list(valid[0][0][0].columns)

    X_runs, y_runs, run_names = [], [], []
    for (df, y), fname in valid:
        if list(df.columns) != feature_names:
            raise ValueError("CSV 파일들 간 컬럼 구성이 다릅니다.")
        X_runs.append(df.to_numpy(dtype=np.float32))
        y_runs.append(y)
        run_names.append(fname)

    # 쓸모없는 변수 제거
    if exclude_useless_features:
        useless_features_json = _resolve_useless_json_path(useless_features_json)
        if os.path.exists(useless_features_json):
            with open(useless_features_json, 'r') as f:
                useless_data = json.load(f)
            useless_list = useless_data.get("useless_features", [])

            remove_indices = [i for i, name in enumerate(feature_names) if name in useless_list]
            keep_indices = [i for i in range(len(feature_names)) if i not in remove_indices]

            if verbose:
                print(f"\nRemoving {len(remove_indices)} useless features")
                print(f"   Features: {len(feature_names)} -> {len(keep_indices)}")
                print(f"   Resolved path: {useless_features_json}")

            X_runs = [X[:, keep_indices] for X in X_runs]
            feature_names = [feature_names[i] for i in keep_indices]
        else:
            if verbose:
                print(f"\nUseless features JSON not found: {useless_features_json}")

    if verbose:
        print("\n[Data Class Distribution - Run-based]")
        print(f"Total runs: {len(X_runs)}")
        total_samples = sum(len(X) for X in X_runs)
        print(f"Total samples: {total_samples}")

        label_counts = {}
        for y_r in y_runs:
            label = int(y_r[0])
            label_counts[label] = label_counts.get(label, 0) + len(y_r)

        for lid in sorted(label_counts.keys()):
            class_name = ID2LABEL.get(lid, f"UNKNOWN({lid})")
            count = label_counts[lid]
            print(f"  {class_name:15s}: {count:6d} samples ({sum(1 for yr in y_runs if int(yr[0]) == lid)} runs)")

        missing_classes = []
        for i, label_name in enumerate(LABELS):
            if i not in label_counts:
                missing_classes.append(label_name)
        if missing_classes:
            print(f"\n  Missing classes: {', '.join(missing_classes)}")

    return X_runs, y_runs, run_names, feature_names


def create_sliding_windows_from_runs(X_runs, y_runs, window_size, stride=1):
    """
    런 단위로 슬라이딩 윈도우 생성 (파일 경계 보존).
    stride로 이동 간격 조절 가능.

    Args:
        X_runs: List[np.ndarray] - 각 (N_i, D)
        y_runs: List[np.ndarray] - 각 (N_i,)
        window_size: int
        stride: int - 윈도우 이동 간격 (1=매 샘플, 2=1초 간격 if 0.5초 샘플링)

    Returns:
        X_windows: (총 윈도우 수, window_size, D)
        y_windows: (총 윈도우 수,) - 마지막 타임스텝 라벨
    """
    all_X, all_y = [], []

    for X_run, y_run in zip(X_runs, y_runs):
        N = X_run.shape[0]
        for i in range(0, N - window_size + 1, stride):
            all_X.append(X_run[i : i + window_size])
            all_y.append(y_run[i + window_size - 1])

    if len(all_X) == 0:
        raise ValueError("윈도우를 생성할 수 없습니다. 런 길이가 window_size보다 짧습니다.")

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.int64)


