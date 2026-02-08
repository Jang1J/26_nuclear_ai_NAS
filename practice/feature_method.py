"""
Physics 기반 피처 엔지니어링.

원본 179개 피처에 도메인 지식 기반 28개 물리 피처를 추가하여 207개 피처 생성.
- 1차/2차 미분 (8+8 = 16개)
- 루프 간 비대칭 (7개)
- 물리적 상관관계 커플링 (5개)
"""
import numpy as np


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
        """이동 표준편차 (벡터화 구현, H6 fix)."""
        import pandas as pd
        result = pd.Series(arr).rolling(
            window=self.moving_std_window, min_periods=2
        ).std().fillna(0.0).values.astype(np.float32)
        return result


def make_feature_method(name="physics", **kwargs):
    """Physics 피처 엔지니어링 생성."""
    if name == "physics":
        return FeaturePhysics(**kwargs)
    raise ValueError(f"unknown feature method: {name} (only 'physics' is supported)")
