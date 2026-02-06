import os
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None


class FeatureAll:
    def fit(self, X, y, feature_names):
        self.feature_names_all = list(feature_names)
        self.feature_names = list(feature_names)
        return self

    def transform(self, X, y):
        return X, y, self.feature_names

    def transform_runs(self, X_runs, y_runs):
        return X_runs, y_runs, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeatureChangeOnly:
    def fit(self, X, y, feature_names):
        diff = X.max(axis=0) - X.min(axis=0)
        self.keep_idx = np.where(diff > 0)[0]
        self.feature_names_all = list(feature_names)
        self.feature_names = [feature_names[i] for i in self.keep_idx]
        return self

    def transform(self, X, y):
        return X[:, self.keep_idx], y, self.feature_names

    def transform_runs(self, X_runs, y_runs):
        out_X = [X[:, self.keep_idx] for X in X_runs]
        return out_X, y_runs, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeatureSelectionLGBM:
    """LightGBM importance 기반 Top-K 선택."""
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
            raise ImportError("lightgbm이 설치되어 있지 않습니다. pip install lightgbm")

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

        order = np.argsort(-imp)
        k = min(self.topk, len(order))
        self.keep_idx = order[:k]
        self.feature_names = [feature_names[i] for i in self.keep_idx]

        if self.save_model:
            import joblib
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            joblib.dump(
                {"model": self.model, "keep_idx": self.keep_idx,
                 "feature_names_all": self.feature_names_all,
                 "feature_names": self.feature_names},
                self.model_path,
            )

        if self.topk_plot_path is not None:
            try:
                import matplotlib.pyplot as plt
                os.makedirs(os.path.dirname(self.topk_plot_path) or ".", exist_ok=True)
                top_imp = imp[self.keep_idx]
                names = [feature_names[i] for i in self.keep_idx]
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

    def transform_runs(self, X_runs, y_runs):
        out_X = [X[:, self.keep_idx] for X in X_runs]
        return out_X, y_runs, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeatureWithDiff:
    """원본 + 1차 차분. 런 단위로 적용하면 경계 누수 없음."""
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

    def transform_runs(self, X_runs, y_runs):
        """런 단위로 diff 계산 -> 경계 누수 없음"""
        out_X = []
        for X_run in X_runs:
            diff = np.zeros_like(X_run)
            diff[1:] = X_run[1:] - X_run[:-1]
            out_X.append(np.hstack([X_run, diff]))
        return out_X, y_runs, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeatureWithStats:
    """원본 + 이동평균 + 이동표준편차. 런 단위 적용 지원."""
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

    def transform_runs(self, X_runs, y_runs):
        """런 단위로 rolling 계산 -> 경계 누수 없음"""
        import pandas as pd
        out_X = []
        for X_run in X_runs:
            df = pd.DataFrame(X_run)
            rolling = df.rolling(window=self.stat_window, min_periods=1)
            rmean = rolling.mean().values.astype(np.float32)
            rstd = rolling.std().fillna(0).values.astype(np.float32)
            out_X.append(np.hstack([X_run, rmean, rstd]))
        return out_X, y_runs, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)


class FeaturePhysics:
    """
    원전 도메인 지식 기반 물리 피처 28개 생성.
    런 단위 적용 시 diff가 런 경계 내에서만 계산됨.
    """

    _DERIV_TARGETS = [
        "PPRZ", "PSG1", "PSG2", "PSG3",
        "PCTMT", "ZSGNOR1", "ZSGNOR2", "ZSGNOR3",
    ]
    _SG_PRESSURE = ["PSG1", "PSG2", "PSG3"]
    _SG_LEVEL = ["ZSGNOR1", "ZSGNOR2", "ZSGNOR3"]
    _EXTRA_COLS = ["UCTMT", "ZPRZ"]

    def __init__(self, moving_std_window=3, **kwargs):
        self.moving_std_window = moving_std_window

    def fit(self, X, y, feature_names):
        self.feature_names_all = list(feature_names)
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        all_required_cols = set(self._DERIV_TARGETS + self._SG_PRESSURE + self._SG_LEVEL + self._EXTRA_COLS)
        missing_cols = [col for col in all_required_cols if col not in name_to_idx]

        if missing_cols:
            raise ValueError(
                f"Physics 피처에 필요한 컬럼이 없습니다: {missing_cols}\n"
                f"현재 컬럼: {feature_names[:10]}... (총 {len(feature_names)}개)"
            )

        self._deriv_idx = [name_to_idx[c] for c in self._DERIV_TARGETS]
        self._sg_p_idx = [name_to_idx[c] for c in self._SG_PRESSURE]
        self._sg_l_idx = [name_to_idx[c] for c in self._SG_LEVEL]
        self._idx_PPRZ = name_to_idx["PPRZ"]
        self._idx_UCTMT = name_to_idx["UCTMT"]
        self._idx_ZPRZ = name_to_idx["ZPRZ"]
        self._idx_PCTMT = name_to_idx["PCTMT"]

        d1_names = [f"d{c}" for c in self._DERIV_TARGETS]
        d2_names = [f"d2_{c}" for c in self._DERIV_TARGETS]
        asym_names = [
            "PSG_range", "ZSGNOR_range",
            "PSG1_ratio", "PSG2_ratio", "PSG3_ratio",
            "PSG_std", "ZSGNOR_std",
        ]
        coupling_names = [
            "PPRZ-PSGavg", "dPCTMTx dUCTMT", "MovStd(PPRZ)",
            "PPRZxZPRZ", "dPPRZx dPSGavg",
        ]

        self.feature_names = (
            list(feature_names)
            + d1_names + d2_names
            + asym_names + coupling_names
        )
        return self

    def _compute_physics(self, X):
        """단일 배열에 대해 물리 피처 계산 (런 1개 or 전체)"""
        N, D = X.shape

        # 1차/2차 미분
        diff1 = np.zeros_like(X)
        diff1[1:] = X[1:] - X[:-1]
        diff2 = np.zeros_like(X)
        diff2[2:] = diff1[2:] - diff1[1:-1]

        d1_features = diff1[:, self._deriv_idx]
        d2_features = diff2[:, self._deriv_idx]

        # 루프 간 비대칭
        psg = X[:, self._sg_p_idx]
        zsgnor = X[:, self._sg_l_idx]

        psg_range = (psg.max(axis=1) - psg.min(axis=1)).reshape(-1, 1)
        zsgnor_range = (zsgnor.max(axis=1) - zsgnor.min(axis=1)).reshape(-1, 1)

        psg_sum = psg.sum(axis=1, keepdims=True) + 1e-8
        psg_ratios = psg / psg_sum
        psg_std = psg.std(axis=1).reshape(-1, 1)
        zsgnor_std = zsgnor.std(axis=1).reshape(-1, 1)

        asym_features = np.hstack([
            psg_range, zsgnor_range, psg_ratios,
            psg_std, zsgnor_std,
        ])

        # 물리적 상관관계
        pprz = X[:, self._idx_PPRZ]
        psg_mean = psg.mean(axis=1)
        zprz = X[:, self._idx_ZPRZ]

        d_PPRZ = diff1[:, self._idx_PPRZ]
        d_PCTMT = diff1[:, self._idx_PCTMT]
        d_UCTMT = diff1[:, self._idx_UCTMT]
        d_psg_mean = diff1[:, self._sg_p_idx].mean(axis=1)

        c1 = (pprz - psg_mean).reshape(-1, 1)
        c2 = (d_PCTMT * d_UCTMT).reshape(-1, 1)
        c3 = self._moving_std(pprz).reshape(-1, 1)
        c4 = (pprz * zprz).reshape(-1, 1)
        c5 = (d_PPRZ * d_psg_mean).reshape(-1, 1)

        coupling_features = np.hstack([c1, c2, c3, c4, c5])

        X_aug = np.hstack([
            X, d1_features, d2_features,
            asym_features, coupling_features,
        ]).astype(np.float32)

        return X_aug

    def transform(self, X, y):
        return self._compute_physics(X), y, self.feature_names

    def transform_runs(self, X_runs, y_runs):
        """런 단위로 물리 피처 계산 -> diff가 런 경계를 넘지 않음"""
        out_X = [self._compute_physics(X_run) for X_run in X_runs]
        return out_X, y_runs, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)

    def _moving_std(self, arr):
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
