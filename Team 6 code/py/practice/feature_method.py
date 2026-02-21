"""
Physics 기반 피처 엔지니어링.

원본 피처에 도메인 지식 기반 물리 피처를 추가.
- 1차/2차 미분 (8+8 = 16개)
- 루프 간 비대칭 (7개)  — PSG, ZSGNOR
- 물리적 상관관계 커플링 (5개)
[Phase 1 확장]
- 추가 비대칭 3그룹 (15개) — UHOLEG, ZSGN, WSTM
- 정규화 비대칭 (2개)   — 운전점 보정
- 핵심 미분 추가 (3개)  — dVSUMP, d(PSG_range)/dt, d(ZSGNOR_range)/dt
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

    # Phase 1: 추가 비대칭 그룹
    _HOTLEG_TEMP = ["UHOLEG1", "UHOLEG2", "UHOLEG3"]   # Hot leg 온도 (129배 증가)
    _SG_NARROW = ["ZSGN1", "ZSGN2", "ZSGN3"]           # SG Narrow 레벨 (818배 증가)
    _STEAM_FLOW = ["WSTM1", "WSTM2", "WSTM3"]          # Steam 유량 (10배 증가)
    _PHASE1_COLS = ["VSUMP"]                             # 섬프 수위 (LOCA 핵심)

    def __init__(self, moving_std_window=3, **kwargs):
        self.moving_std_window = moving_std_window

    def fit(self, X, y, feature_names):
        self.feature_names_all = list(feature_names)
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        all_required_cols = set(
            self._DERIV_TARGETS + self._SG_PRESSURE + self._SG_LEVEL + self._EXTRA_COLS
            + self._HOTLEG_TEMP + self._SG_NARROW + self._STEAM_FLOW + self._PHASE1_COLS
        )
        missing_cols = [col for col in all_required_cols if col not in name_to_idx]

        if missing_cols:
            raise ValueError(
                f"Physics 피처에 필요한 컬럼이 없습니다: {missing_cols}\n"
                f"현재 컬럼: {feature_names[:10]}... (총 {len(feature_names)}개)"
            )

        # 기존 인덱스
        self._deriv_idx = [name_to_idx[c] for c in self._DERIV_TARGETS]
        self._sg_p_idx = [name_to_idx[c] for c in self._SG_PRESSURE]
        self._sg_l_idx = [name_to_idx[c] for c in self._SG_LEVEL]
        self._idx_PPRZ = name_to_idx["PPRZ"]
        self._idx_UCTMT = name_to_idx["UCTMT"]
        self._idx_ZPRZ = name_to_idx["ZPRZ"]
        self._idx_PCTMT = name_to_idx["PCTMT"]

        # Phase 1 인덱스
        self._hotleg_idx = [name_to_idx[c] for c in self._HOTLEG_TEMP]
        self._sgn_idx = [name_to_idx[c] for c in self._SG_NARROW]
        self._steam_idx = [name_to_idx[c] for c in self._STEAM_FLOW]
        self._idx_VSUMP = name_to_idx["VSUMP"]

        # ── Feature 이름 정의 ──
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

        # Phase 1 이름
        asym_hotleg_names = [
            "UHOLEG_range", "UHOLEG_std",
            "UHOLEG1_ratio", "UHOLEG2_ratio", "UHOLEG3_ratio",
        ]
        asym_sgn_names = [
            "ZSGN_range", "ZSGN_std",
            "ZSGN1_ratio", "ZSGN2_ratio", "ZSGN3_ratio",
        ]
        asym_steam_names = [
            "WSTM_range", "WSTM_std",
            "WSTM1_ratio", "WSTM2_ratio", "WSTM3_ratio",
        ]
        norm_asym_names = ["PSG_norm_asym", "ZSGNOR_norm_asym"]
        extra_deriv_names = ["dVSUMP", "d(PSG_range)/dt", "d(ZSGNOR_range)/dt"]

        self.feature_names = (
            list(feature_names)
            + d1_names + d2_names
            + asym_names + coupling_names
            + asym_hotleg_names + asym_sgn_names + asym_steam_names
            + norm_asym_names + extra_deriv_names
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

        # ══════════════════════════════════════════
        # Phase 1: 추가 비대칭 3그룹 (+15개)
        # ══════════════════════════════════════════
        # UHOLEG 비대칭 (Hot leg 온도, SGTR시 129배 증가)
        uholeg = X[:, self._hotleg_idx]
        uholeg_range = (uholeg.max(axis=1) - uholeg.min(axis=1)).reshape(-1, 1)
        uholeg_std = uholeg.std(axis=1).reshape(-1, 1)
        uholeg_sum = uholeg.sum(axis=1, keepdims=True) + 1e-8
        uholeg_ratios = uholeg / uholeg_sum

        # ZSGN 비대칭 (SG Narrow 레벨, SGTR시 818배 증가)
        zsgn = X[:, self._sgn_idx]
        zsgn_range = (zsgn.max(axis=1) - zsgn.min(axis=1)).reshape(-1, 1)
        zsgn_std = zsgn.std(axis=1).reshape(-1, 1)
        zsgn_sum = zsgn.sum(axis=1, keepdims=True) + 1e-8
        zsgn_ratios = zsgn / zsgn_sum

        # WSTM 비대칭 (Steam 유량, SGTR시 10배 증가)
        wstm = X[:, self._steam_idx]
        wstm_range = (wstm.max(axis=1) - wstm.min(axis=1)).reshape(-1, 1)
        wstm_std = wstm.std(axis=1).reshape(-1, 1)
        wstm_sum = wstm.sum(axis=1, keepdims=True) + 1e-8
        wstm_ratios = wstm / wstm_sum

        phase1_asym = np.hstack([
            uholeg_range, uholeg_std, uholeg_ratios,   # 5개
            zsgn_range, zsgn_std, zsgn_ratios,          # 5개
            wstm_range, wstm_std, wstm_ratios,          # 5개
        ])

        # ══════════════════════════════════════════
        # Phase 1: 정규화 비대칭 (+2개)
        # ══════════════════════════════════════════
        psg_avg = psg.mean(axis=1, keepdims=True) + 1e-8
        psg_norm_dev = (psg - psg_avg) / psg_avg
        psg_norm_asym = psg_norm_dev.std(axis=1).reshape(-1, 1)

        zsgnor_avg = zsgnor.mean(axis=1, keepdims=True) + 1e-8
        zsgnor_norm_dev = (zsgnor - zsgnor_avg) / zsgnor_avg
        zsgnor_norm_asym = zsgnor_norm_dev.std(axis=1).reshape(-1, 1)

        phase1_norm_asym = np.hstack([psg_norm_asym, zsgnor_norm_asym])

        # ══════════════════════════════════════════
        # Phase 1: 핵심 미분 추가 (+3개)
        # ══════════════════════════════════════════
        # dVSUMP: 섬프 수위 변화율 (LOCA 핵심)
        d_VSUMP = diff1[:, self._idx_VSUMP].reshape(-1, 1)

        # d(PSG_range)/dt: 비대칭 확대 속도
        d_psg_range = np.zeros_like(psg_range)
        d_psg_range[1:] = psg_range[1:] - psg_range[:-1]

        # d(ZSGNOR_range)/dt: SG 레벨 비대칭 확대 속도
        d_zsgnor_range = np.zeros_like(zsgnor_range)
        d_zsgnor_range[1:] = zsgnor_range[1:] - zsgnor_range[:-1]

        phase1_deriv = np.hstack([d_VSUMP, d_psg_range, d_zsgnor_range])

        # ══════════════════════════════════════════
        # 최종 결합
        # ══════════════════════════════════════════
        X_aug = np.hstack([
            X, d1_features, d2_features,
            asym_features, coupling_features,
            phase1_asym, phase1_norm_asym, phase1_deriv,
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
