import os
import re
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from multiprocessing import Pool

# 우리가 최종적으로 분류할 9개 클래스
LABELS = [
    "NORMAL",        # 0: 정상
    "LOCA_HL",       # 1: Hot Leg LOCA
    "LOCA_CL",       # 2: Cold Leg LOCA
    "LOCA_RCPCSEAL", # 3: RCP Seal LOCA
    "SGTR_SG1",      # 4: 증기발생기 1번 파열
    "SGTR_SG2",      # 5: 증기발생기 2번 파열
    "SGTR_SG3",      # 6: 증기발생기 3번 파열
    "MSLB_in",       # 7: 주증기관 파열 (내부)
    "MSLB_out"       # 8: 주증기관 파열 (외부)
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


def infer_label_id(file_name: str):
    """
    파일명 prefix로 라벨 id 추론 (9개 클래스).
    지원하는 예:
      - NORMAL_*.csv
      - LOCA_HL_*.csv (Hot Leg LOCA)
      - LOCA_CL_*.csv (Cold Leg LOCA)
      - LOCA_RCPCSEAL_*.csv (RCP Seal LOCA)
      - SGTR_SG1_*.csv (증기발생기 1번)
      - SGTR_SG2_*.csv (증기발생기 2번)
      - SGTR_SG3_*.csv (증기발생기 3번)
      - MSLBIN_*.csv (MSLB Inside)
      - MSLBOUT_*.csv (MSLB Outside)

    현재 데이터: LOCA_HL(0개), LOCA_RCPCSEAL(0개) 제외한 7개 클래스
    """
    base = os.path.basename(file_name)

    # collect_NORMAL_... 은 NORMAL로 취급
    if base.startswith("collect_NORMAL_"):
        return LABEL2ID["NORMAL"]

    # 일반 prefix
    if base.startswith("NORMAL_"):
        return LABEL2ID["NORMAL"]

    # LOCA 세분화 (Hot Leg, Cold Leg, RCP Seal)
    if base.startswith("LOCA_HL_"):
        return LABEL2ID["LOCA_HL"]
    if base.startswith("LOCA_CL_"):
        return LABEL2ID["LOCA_CL"]
    if base.startswith("LOCA_RCPCSEAL_"):
        return LABEL2ID["LOCA_RCPCSEAL"]

    # SGTR 세분화 (SG1, SG2, SG3)
    if base.startswith("SGTR_SG1_"):
        return LABEL2ID["SGTR_SG1"]
    if base.startswith("SGTR_SG2_"):
        return LABEL2ID["SGTR_SG2"]
    if base.startswith("SGTR_SG3_"):
        return LABEL2ID["SGTR_SG3"]

    # MSLB (Inside, Outside)
    if base.startswith("MSLBIN_") or base.startswith("MSLB_in_"):
        return LABEL2ID["MSLB_in"]
    if base.startswith("MSLBOUT_") or base.startswith("MSLB_out_"):
        return LABEL2ID["MSLB_out"]

    # ⚠️ 이전 형식 호환성: 세부 분류되지 않은 데이터
    # 파일명에서 자동으로 LOCA_CL로 매핑 (현재 데이터가 모두 Cold Leg인 경우)
    if base.startswith("LOCA_"):
        # 향후 데이터가 세부 분류되면 이 경고가 사라질 것
        return LABEL2ID["LOCA_CL"]

    # SGTR도 마찬가지로 세부 분류되지 않은 경우
    # 실제 데이터를 보고 SG1/SG2/SG3 중 적절히 매핑 필요
    # 임시로 SG1로 매핑
    if base.startswith("SGTR_"):
        return LABEL2ID["SGTR_SG1"]

    return None


def extract_number(fname: str) -> int:
    """
    파일명 끝의 번호를 정렬키로 쓰기.
    예: LOCA_100100_10_1.csv -> 1
        MSLBIN_120100_10_3+.csv 처럼 이상한 문자가 있어도 마지막 숫자를 잡아줌.
    """
    base = os.path.basename(fname)
    # 마지막에 등장하는 숫자들 중 맨 끝 숫자 추출
    nums = re.findall(r"(\d+)", base)
    return int(nums[-1]) if nums else -1


def load_one_csv(file_name: str, folder_path: str, include_time: bool = False):
    label_id = infer_label_id(file_name)
    if label_id is None:
        return None

    fp = os.path.join(folder_path, file_name)
    df = pd.read_csv(fp)

    # 시간 컬럼이 KCNTOMS 로 들어오는 경우가 많음 (스크린샷 기준)
    if (not include_time) and ("KCNTOMS" in df.columns):
        df = df.drop(columns=["KCNTOMS"])

    y = np.full((len(df),), label_id, dtype=np.int64)
    return df, y


def load_Xy(folder_path: str, include_time: bool = False, n_workers=None, verbose=True):
    """
    폴더 전체의 CSV를 읽어 (X,y,feature_names) 반환.
    - 파일 리스트는 (라벨 순서, 번호 순서) 로 정렬
    - multiprocessing 사용
    - verbose=True일 때 실제 클래스 분포 출력
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

    results = [r for r in results if r is not None]
    if len(results) == 0:
        raise RuntimeError("라벨(prefix)과 매칭되는 CSV가 없습니다. 파일명 prefix를 확인하세요.")

    feature_names = list(results[0][0].columns)

    X_list, y_list = [], []
    for df, y in results:
        if list(df.columns) != feature_names:
            raise ValueError("CSV 파일들 간 컬럼 구성이 다릅니다.")
        X_list.append(df.to_numpy(dtype=np.float32))
        y_list.append(y)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 실제 데이터 클래스 분포 출력
    if verbose:
        print("\n[Data Class Distribution]")
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Total samples: {len(y)}")
        for label_id, count in zip(unique_labels, counts):
            class_name = ID2LABEL.get(int(label_id), f"UNKNOWN({label_id})")
            print(f"  {class_name:15s}: {count:6d} samples")

        # 누락된 클래스 경고
        missing_classes = []
        for i, label_name in enumerate(LABELS):
            if i not in unique_labels:
                missing_classes.append(label_name)

        if missing_classes:
            print(f"\n⚠️  Missing classes in current data: {', '.join(missing_classes)}")
            print("   (This is OK if you plan to add them later)")

    return X, y, feature_names


def create_sliding_windows_grouped(X, y, window_size, group_size):
    """
    그룹 경계를 넘지 않는 슬라이딩 윈도우 생성.

    split 후 데이터는 셔플된 그룹들의 flat 배열이므로,
    그룹 내부에서만 윈도우를 만들어야 의미 있는 시계열이 됨.

    Parameters:
        X: (N, D) — N 타임스텝, D 피처
        y: (N,) — 라벨
        window_size: 윈도우 크기 (≤ group_size)
        group_size: 그룹 크기 (data_split에서 사용한 값과 동일해야 함)

    Returns:
        X_win: (n_windows, window_size, D)
        y_win: (n_windows,) — 각 윈도우 마지막 타임스텝의 라벨
    """
    N, D = X.shape
    n_groups = N // group_size
    X_trimmed = X[: n_groups * group_size].reshape(n_groups, group_size, D)
    y_trimmed = y[: n_groups * group_size].reshape(n_groups, group_size)

    windows_per_group = group_size - window_size + 1
    X_wins = []
    y_wins = []

    for g in range(n_groups):
        for i in range(windows_per_group):
            X_wins.append(X_trimmed[g, i : i + window_size, :])
            y_wins.append(y_trimmed[g, i + window_size - 1])

    return np.array(X_wins, dtype=X.dtype), np.array(y_wins, dtype=y.dtype)


if __name__ == "__main__":
    folder = "data/data_new"  # 예시
    X, y, feature_names = load_Xy(folder, include_time=False)
    print("[dataloader]")
    print("X:", X.shape[1:])
    print("y:", y.shape[1:], "labels:", np.unique(y))
    print("num_features:", len(feature_names))
    print("first 20 feature names:", feature_names[:20])
    print("first 20 samples X:\n", X[:20])
    print("first 2 files y:\n", y[:120])
    print("last 2 files y:\n", y[-120:])