import os
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None

from .dataloader import ID2LABEL


class FeatureAll:
    def fit(self, X, y, feature_names):
        self.feature_names_all = list(feature_names)
        self.feature_names = list(feature_names)
        return self

    def transform(self, X, y):
        return X, y, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeatureChangeOnly:
    # max-min이 0보다 큰 변수만 사용
    def fit(self, X, y, feature_names):
        diff = X.max(axis=0) - X.min(axis=0)
        self.keep_idx = np.where(diff > 0)[0]
        self.feature_names_all = list(feature_names)
        self.feature_names = [feature_names[i] for i in self.keep_idx]
        return self

    def transform(self, X, y):
        return X[:, self.keep_idx], y, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeatureSelectionLGBM:
    """
    LightGBM importance 기반 Top-K 선택.
    - 학습 후 importance plot 저장 옵션 (필요시)
    """
    def __init__(
        self,
        seed=0,
        model_path="feature_selector_lgbm.pkl",
        save_model=True,
        importance_type="split",
        topk=300,
        topk_plot_path=None,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
    ):
        self.seed = seed
        self.model_path = model_path
        self.save_model = save_model
        self.importance_type = importance_type
        self.topk = topk
        self.topk_plot_path = topk_plot_path

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.model = None
        self.keep_idx = None
        self.feature_names_all = None
        self.feature_names = None

    def fit(self, X, y, feature_names):
        if lgb is None:
            raise ImportError(
                "lightgbm 이 설치되어 있지 않습니다.\n"
                "설치: pip install lightgbm\n"
                "또는 feature_method=all/change 로 먼저 진행하세요."
            )

        self.feature_names_all = list(feature_names)

        self.model = lgb.LGBMClassifier(
            random_state=self.seed,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=-1,
        )
        self.model.fit(X, y)

        imp = self.model.booster_.feature_importance(importance_type=self.importance_type)
        imp = np.asarray(imp, dtype=np.float64)

        order = np.argsort(-imp)  # 내림차순
        k = min(self.topk, len(order))
        self.keep_idx = order[:k]
        self.feature_names = [feature_names[i] for i in self.keep_idx]

        if self.save_model:
            import joblib
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            joblib.dump(
                {
                    "model": self.model,
                    "keep_idx": self.keep_idx,
                    "feature_names_all": self.feature_names_all,
                    "feature_names": self.feature_names,
                },
                self.model_path,
            )

        # plot은 utils_plot 없으면 여기서 간단히 처리
        if self.topk_plot_path is not None:
            try:
                import matplotlib.pyplot as plt
                os.makedirs(os.path.dirname(self.topk_plot_path) or ".", exist_ok=True)
                top_imp = imp[self.keep_idx]
                names = [feature_names[i] for i in self.keep_idx]
                # 너무 길면 상위 20만
                n_show = min(20, len(names))
                plt.figure()
                plt.barh(range(n_show)[::-1], top_imp[:n_show][::-1])
                plt.yticks(range(n_show)[::-1], names[:n_show][::-1], fontsize=8)
                plt.title("Top feature importance")
                plt.tight_layout()
                plt.savefig(self.topk_plot_path, dpi=150)
                plt.close()
            except Exception:
                pass

        return self

    def transform(self, X, y):
        return X[:, self.keep_idx], y, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeatureWithDiff:
    """
    원본 피처에 1차 차분(변화율)을 추가.
    사고 발생 시 센서값의 급격한 변화를 포착하는 데 유용.
    출력 차원: D' = D_original * 2
    """
    def fit(self, X, y, feature_names):
        self.feature_names_all = list(feature_names)
        diff_names = [f"{fn}_diff" for fn in feature_names]
        self.feature_names = list(feature_names) + diff_names
        return self

    def transform(self, X, y):
        diff = np.zeros_like(X)
        diff[1:] = X[1:] - X[:-1]
        X_aug = np.hstack([X, diff])
        return X_aug, y, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeatureWithStats:
    """
    원본 피처에 이동평균 + 이동표준편차를 추가.
    노이즈 제거 및 트렌드 파악에 유용.
    출력 차원: D' = D_original * 3
    """
    def __init__(self, stat_window=5):
        self.stat_window = stat_window

    def fit(self, X, y, feature_names):
        self.feature_names_all = list(feature_names)
        mean_names = [f"{fn}_rmean{self.stat_window}" for fn in feature_names]
        std_names = [f"{fn}_rstd{self.stat_window}" for fn in feature_names]
        self.feature_names = list(feature_names) + mean_names + std_names
        return self

    def transform(self, X, y):
        import pandas as pd
        df = pd.DataFrame(X)
        rolling = df.rolling(window=self.stat_window, min_periods=1)
        rmean = rolling.mean().values.astype(np.float32)
        rstd = rolling.std().fillna(0).values.astype(np.float32)
        X_aug = np.hstack([X, rmean, rstd])
        return X_aug, y, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeaturePhysics:
    """
    원전 도메인 지식 기반 물리 피처 엔지니어링.

    3가지 카테고리의 파생 피처 28개를 생성:
      1) 시계열 미분 (d/dt, d²/dt²) — 사고 초기 변화 감지  (16개)
      2) 루프 간 비대칭 — SGTR/MSLB 파손 루프 식별         (7개)
      3) 물리적 상관관계 — 사고 유형 판별 증폭              (5개)

    Parameters:
        group_size: int — 그룹 크기. 미분 계산 시 그룹 경계에서 0 처리 (기본 10)
        moving_std_window: int — PPRZ 이동표준편차 윈도우 (기본 3)
    """

    # 1차 미분 대상 센서 컬럼명
    _DERIV_TARGETS = [
        "PPRZ", "PSG1", "PSG2", "PSG3",
        "PCTMT", "ZSGNOR1", "ZSGNOR2", "ZSGNOR3",
    ]
    # SG 루프 컬럼 (비대칭 계산용)
    _SG_PRESSURE = ["PSG1", "PSG2", "PSG3"]
    _SG_LEVEL = ["ZSGNOR1", "ZSGNOR2", "ZSGNOR3"]
    # 물리적 상관관계용 추가 컬럼
    _EXTRA_COLS = ["UCTMT", "ZPRZ"]

    def __init__(self, group_size=10, moving_std_window=3):
        self.group_size = group_size
        self.moving_std_window = moving_std_window

    def fit(self, X, y, feature_names):
        self.feature_names_all = list(feature_names)
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        # 미분 대상 인덱스
        self._deriv_idx = [name_to_idx[c] for c in self._DERIV_TARGETS]

        # SG 루프 인덱스
        self._sg_p_idx = [name_to_idx[c] for c in self._SG_PRESSURE]
        self._sg_l_idx = [name_to_idx[c] for c in self._SG_LEVEL]

        # 추가 컬럼 인덱스
        self._idx_PPRZ = name_to_idx["PPRZ"]
        self._idx_UCTMT = name_to_idx["UCTMT"]
        self._idx_ZPRZ = name_to_idx["ZPRZ"]
        self._idx_PCTMT = name_to_idx["PCTMT"]

        # 출력 피처 이름 구성 (db.txt 네이밍과 동일)
        # 1차 미분 8개: dPPRZ, dPSG1~3, dPCTMT, dZSGNOR1~3
        d1_names = [f"d{c}" for c in self._DERIV_TARGETS]            # 8개
        # 2차 미분 8개: d2_PPRZ, d2_PSG1~3, d2_PCTMT, d2_ZSGNOR1~3
        d2_names = [f"d2_{c}" for c in self._DERIV_TARGETS]         # 8개

        # 비대칭 7개 (db.txt 네이밍과 동일)
        asym_names = [
            "PSG_range", "ZSGNOR_range",
            "PSG1_ratio", "PSG2_ratio", "PSG3_ratio",
            "PSG_std", "ZSGNOR_std",
        ]                                                            # 7개

        # 물리적 상관관계 5개 (db.txt 네이밍과 동일)
        coupling_names = [
            "PPRZ-PSGavg",        # 1차-2차 압력차
            "dPCTMTx dUCTMT",     # dPCTMT × dUCTMT
            "MovStd(PPRZ)",       # PPRZ 이동표준편차
            "PPRZxZPRZ",          # PPRZ × ZPRZ
            "dPPRZx dPSGavg",     # dPPRZ × dPSGavg
        ]                                                            # 5개

        self.feature_names = (
            list(feature_names)
            + d1_names + d2_names
            + asym_names + coupling_names
        )
        return self

    def transform(self, X, y):
        N, D = X.shape
        G = self.group_size

        # ── 1) 그룹 경계 안전한 1차·2차 미분 ──
        n_groups = N // G
        n_used = n_groups * G

        # 그룹 단위 reshape → 그룹 내에서만 diff
        X_grp = X[:n_used].reshape(n_groups, G, D)
        diff1_grp = np.zeros_like(X_grp)
        diff1_grp[:, 1:, :] = X_grp[:, 1:, :] - X_grp[:, :-1, :]

        diff2_grp = np.zeros_like(X_grp)
        diff2_grp[:, 2:, :] = diff1_grp[:, 2:, :] - diff1_grp[:, 1:-1, :]

        diff1_flat = diff1_grp.reshape(n_used, D)
        diff2_flat = diff2_grp.reshape(n_used, D)

        # 나머지 행 (N이 G로 나누어떨어지지 않을 때)
        if n_used < N:
            rem = N - n_used
            d1_rem = np.zeros((rem, D), dtype=X.dtype)
            d2_rem = np.zeros((rem, D), dtype=X.dtype)
            if rem > 1:
                d1_rem[1:] = X[n_used + 1:] - X[n_used:-1]
            if rem > 2:
                d2_rem[2:] = d1_rem[2:] - d1_rem[1:-1]
            diff1_flat = np.vstack([diff1_flat, d1_rem])
            diff2_flat = np.vstack([diff2_flat, d2_rem])

        # 대상 컬럼만 추출
        d1_features = diff1_flat[:, self._deriv_idx]   # (N, 8)
        d2_features = diff2_flat[:, self._deriv_idx]   # (N, 8)

        # ── 2) 루프 간 비대칭 ──
        psg = X[:, self._sg_p_idx]                     # (N, 3)
        zsgnor = X[:, self._sg_l_idx]                  # (N, 3)

        psg_range = (psg.max(axis=1) - psg.min(axis=1)).reshape(-1, 1)
        zsgnor_range = (zsgnor.max(axis=1) - zsgnor.min(axis=1)).reshape(-1, 1)

        psg_sum = psg.sum(axis=1, keepdims=True) + 1e-8
        psg_ratios = psg / psg_sum                     # (N, 3)

        psg_std = psg.std(axis=1).reshape(-1, 1)
        zsgnor_std = zsgnor.std(axis=1).reshape(-1, 1)

        asym_features = np.hstack([
            psg_range, zsgnor_range,
            psg_ratios,
            psg_std, zsgnor_std,
        ])                                             # (N, 7)

        # ── 3) 물리적 상관관계 ──
        pprz = X[:, self._idx_PPRZ]
        psg_mean = psg.mean(axis=1)
        zprz = X[:, self._idx_ZPRZ]

        d_PPRZ = diff1_flat[:, self._idx_PPRZ]
        d_PCTMT = diff1_flat[:, self._idx_PCTMT]
        d_UCTMT = diff1_flat[:, self._idx_UCTMT]
        d_psg_mean = diff1_flat[:, self._sg_p_idx].mean(axis=1)

        c1 = (pprz - psg_mean).reshape(-1, 1)             # PPRZ-PSGavg
        c2 = (d_PCTMT * d_UCTMT).reshape(-1, 1)           # dPCTMT × dUCTMT
        c3 = self._moving_std(pprz).reshape(-1, 1)        # MovStd(PPRZ)
        c4 = (pprz * zprz).reshape(-1, 1)                 # PPRZ × ZPRZ
        c5 = (d_PPRZ * d_psg_mean).reshape(-1, 1)         # dPPRZ × dPSGavg

        coupling_features = np.hstack([c1, c2, c3, c4, c5])  # (N, 5)

        # ── 결합 ──
        X_aug = np.hstack([
            X, d1_features, d2_features,
            asym_features, coupling_features,
        ]).astype(np.float32)

        return X_aug, y, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)

    def _moving_std(self, arr):
        """이동 표준편차 (min_periods=1, NaN→0)."""
        import pandas as pd
        s = pd.Series(arr)
        return s.rolling(
            window=self.moving_std_window, min_periods=1
        ).std().fillna(0).values.astype(np.float32)


def make_feature_method(name, **kwargs):
    if name == "all":
        return FeatureAll()
    elif name == "change":
        return FeatureChangeOnly()
    elif name == "selection":
        return FeatureSelectionLGBM(**kwargs)
    elif name == "diff":
        return FeatureWithDiff()
    elif name == "stats":
        return FeatureWithStats(**kwargs)
    elif name == "physics":
        return FeaturePhysics(**kwargs)

    raise ValueError(f"unknown feature method: {name}")


if __name__ == "__main__":
    # (시연용) 이 파일만 실행해도 동작 확인 가능
    from .dataloader import load_Xy

    folder = "data/data_new"
    X, y, feature_names = load_Xy(folder, include_time=False)

    _feature_method = "change"  # all / change / selection
    feat = make_feature_method(_feature_method)
    X2, y2, feats2 = feat.fit_transform(X, y, feature_names)

    print("[feature_method]")
    print("features:", len(feature_names), "->", len(feats2))