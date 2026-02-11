"""
Physics 기반 피처 엔지니어링 v4 (No-Derivative).

미분 피처를 완전히 제거하여 시간 간격에 무관한 피처만 사용.
목적: subsample 없이 310행 전체 학습 가능 (1초 간격 → 5배 데이터).

v2(FeaturePhysics) 기반, 미분 제거:
  - 원본 센서값 (188개) — 유지
  - 1차/2차 미분 (16개) — 제거
  - 루프 비대칭 range/std/ratio (7개) — 유지
  - 커플링 c1(PPRZ-PSGavg), c3(MovStd), c4(PPRZxZPRZ) — 유지 (3개)
  - 커플링 c2(dPCTMTxdUCTMT), c5(dPPRZxdPSGavg) — 제거 (2개)
  - Phase1 비대칭 3그룹 (15개) — 유지
  - Phase1 정규화 비대칭 (2개) — 유지
  - Phase1 미분 dVSUMP, d(range)/dt (3개) — 제거

v3 Phase2 피처, 미분 제거:
  - Group1 Pairwise Diff (9개) — 유지
  - Group2 Derivative Diff (6개) — 제거
  - Group3 Deviation from Avg (9개) — 유지
  - Group4 Argmin+Gap (4개) — 유지
  - Group5 Pressurizer Coupling (2개) — 유지

총: 188 + 7 + 3 + 15 + 2 + 9 + 9 + 4 + 2 = 239개 피처
(기존 v3 266개에서 27개 미분 피처 제거)
"""
import numpy as np
import pandas as pd


class FeaturePhysicsV4:
    """
    미분 없는 Physics 피처 엔지니어링.
    시간 간격에 무관한 피처만 사용 → subsample 불필요.
    총 239개 피처 (원본 188 + 물리 51).
    """

    _SG_PRESSURE = ["PSG1", "PSG2", "PSG3"]
    _SG_LEVEL = ["ZSGNOR1", "ZSGNOR2", "ZSGNOR3"]
    _EXTRA_COLS = ["UCTMT", "ZPRZ"]
    _HOTLEG_TEMP = ["UHOLEG1", "UHOLEG2", "UHOLEG3"]
    _SG_NARROW = ["ZSGN1", "ZSGN2", "ZSGN3"]
    _STEAM_FLOW = ["WSTM1", "WSTM2", "WSTM3"]

    def __init__(self, moving_std_window=3, **kwargs):
        self.moving_std_window = moving_std_window

    def fit(self, X, y, feature_names):
        self.feature_names_all = list(feature_names)
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        all_required_cols = set(
            self._SG_PRESSURE + self._SG_LEVEL + self._EXTRA_COLS
            + self._HOTLEG_TEMP + self._SG_NARROW + self._STEAM_FLOW
            + ["PPRZ", "PCTMT"]
        )
        missing_cols = [col for col in all_required_cols if col not in name_to_idx]
        if missing_cols:
            raise ValueError(
                f"Physics 피처에 필요한 컬럼이 없습니다: {missing_cols}\n"
                f"현재 컬럼: {feature_names[:10]}... (총 {len(feature_names)}개)"
            )

        # 인덱스 매핑
        self._sg_p_idx = [name_to_idx[c] for c in self._SG_PRESSURE]
        self._sg_l_idx = [name_to_idx[c] for c in self._SG_LEVEL]
        self._idx_PPRZ = name_to_idx["PPRZ"]
        self._idx_ZPRZ = name_to_idx["ZPRZ"]
        self._hotleg_idx = [name_to_idx[c] for c in self._HOTLEG_TEMP]
        self._sgn_idx = [name_to_idx[c] for c in self._SG_NARROW]
        self._steam_idx = [name_to_idx[c] for c in self._STEAM_FLOW]

        # ── Feature 이름 정의 (미분 제외) ──
        # v2 비대칭 (7개)
        asym_names = [
            "PSG_range", "ZSGNOR_range",
            "PSG1_ratio", "PSG2_ratio", "PSG3_ratio",
            "PSG_std", "ZSGNOR_std",
        ]
        # v2 커플링 — 미분 의존 c2, c5 제거 (3개만)
        coupling_names = [
            "PPRZ-PSGavg", "MovStd(PPRZ)", "PPRZxZPRZ",
        ]
        # Phase1 비대칭 3그룹 (15개)
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
        # Phase1 정규화 비대칭 (2개)
        norm_asym_names = ["PSG_norm_asym", "ZSGNOR_norm_asym"]

        # V3 Phase2 (미분 Group2 제외, 24개)
        phase2_names = [
            # Group1: Pairwise Differences (9개)
            "PSG1-PSG2", "PSG2-PSG3", "PSG1-PSG3",
            "ZSGN1-ZSGN2", "ZSGN2-ZSGN3", "ZSGN1-ZSGN3",
            "UHOLEG1-UHOLEG2", "UHOLEG2-UHOLEG3", "UHOLEG1-UHOLEG3",
            # Group3: Deviation from Average (9개)
            "PSG1-PSGavg", "PSG2-PSGavg", "PSG3-PSGavg",
            "ZSGN1-ZSGNavg", "ZSGN2-ZSGNavg", "ZSGN3-ZSGNavg",
            "UHOLEG1-UHOLEGavg", "UHOLEG2-UHOLEGavg", "UHOLEG3-UHOLEGavg",
            # Group4: Argmin + Gap (4개)
            "PSG_argmin", "PSG_min_gap",
            "ZSGN_argmin", "ZSGN_min_gap",
            # Group5: Pressurizer-SG Coupling (2개)
            "PPRZ_PSGavg_ratio", "PPRZ-PSG2",
        ]

        self.feature_names = (
            list(feature_names)
            + asym_names + coupling_names
            + asym_hotleg_names + asym_sgn_names + asym_steam_names
            + norm_asym_names
            + phase2_names
        )
        return self

    def _compute_physics(self, X):
        """미분 없이 물리 피처 계산."""
        N, D = X.shape

        # ── v2 비대칭 (7개) ──
        psg = X[:, self._sg_p_idx]        # (N, 3)
        zsgnor = X[:, self._sg_l_idx]     # (N, 3)

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

        # ── v2 커플링 (3개, 미분 c2/c5 제거) ──
        pprz = X[:, self._idx_PPRZ]
        psg_mean = psg.mean(axis=1)
        zprz = X[:, self._idx_ZPRZ]

        c1 = (pprz - psg_mean).reshape(-1, 1)               # PPRZ-PSGavg
        c3 = self._moving_std(pprz).reshape(-1, 1)           # MovStd(PPRZ)
        c4 = (pprz * zprz).reshape(-1, 1)                    # PPRZxZPRZ

        coupling_features = np.hstack([c1, c3, c4])

        # ── Phase1 비대칭 3그룹 (15개) ──
        uholeg = X[:, self._hotleg_idx]
        uholeg_range = (uholeg.max(axis=1) - uholeg.min(axis=1)).reshape(-1, 1)
        uholeg_std = uholeg.std(axis=1).reshape(-1, 1)
        uholeg_sum = uholeg.sum(axis=1, keepdims=True) + 1e-8
        uholeg_ratios = uholeg / uholeg_sum

        zsgn = X[:, self._sgn_idx]
        zsgn_range = (zsgn.max(axis=1) - zsgn.min(axis=1)).reshape(-1, 1)
        zsgn_std = zsgn.std(axis=1).reshape(-1, 1)
        zsgn_sum = zsgn.sum(axis=1, keepdims=True) + 1e-8
        zsgn_ratios = zsgn / zsgn_sum

        wstm = X[:, self._steam_idx]
        wstm_range = (wstm.max(axis=1) - wstm.min(axis=1)).reshape(-1, 1)
        wstm_std = wstm.std(axis=1).reshape(-1, 1)
        wstm_sum = wstm.sum(axis=1, keepdims=True) + 1e-8
        wstm_ratios = wstm / wstm_sum

        phase1_asym = np.hstack([
            uholeg_range, uholeg_std, uholeg_ratios,
            zsgn_range, zsgn_std, zsgn_ratios,
            wstm_range, wstm_std, wstm_ratios,
        ])

        # ── Phase1 정규화 비대칭 (2개) ──
        psg_avg = psg.mean(axis=1, keepdims=True) + 1e-8
        psg_norm_dev = (psg - psg_avg) / psg_avg
        psg_norm_asym = psg_norm_dev.std(axis=1).reshape(-1, 1)

        zsgnor_avg = zsgnor.mean(axis=1, keepdims=True) + 1e-8
        zsgnor_norm_dev = (zsgnor - zsgnor_avg) / zsgnor_avg
        zsgnor_norm_asym = zsgnor_norm_dev.std(axis=1).reshape(-1, 1)

        phase1_norm_asym = np.hstack([psg_norm_asym, zsgnor_norm_asym])

        # ── V3 Phase2 (미분 Group2 제외, 24개) ──
        # Group1: Pairwise Differences (9개)
        psg_diff_12 = (psg[:, 0] - psg[:, 1]).reshape(-1, 1)
        psg_diff_23 = (psg[:, 1] - psg[:, 2]).reshape(-1, 1)
        psg_diff_13 = (psg[:, 0] - psg[:, 2]).reshape(-1, 1)

        zsgn_diff_12 = (zsgn[:, 0] - zsgn[:, 1]).reshape(-1, 1)
        zsgn_diff_23 = (zsgn[:, 1] - zsgn[:, 2]).reshape(-1, 1)
        zsgn_diff_13 = (zsgn[:, 0] - zsgn[:, 2]).reshape(-1, 1)

        uholeg_diff_12 = (uholeg[:, 0] - uholeg[:, 1]).reshape(-1, 1)
        uholeg_diff_23 = (uholeg[:, 1] - uholeg[:, 2]).reshape(-1, 1)
        uholeg_diff_13 = (uholeg[:, 0] - uholeg[:, 2]).reshape(-1, 1)

        group1 = np.hstack([
            psg_diff_12, psg_diff_23, psg_diff_13,
            zsgn_diff_12, zsgn_diff_23, zsgn_diff_13,
            uholeg_diff_12, uholeg_diff_23, uholeg_diff_13,
        ])

        # Group3: Deviation from Average (9개)
        psg_avg2 = psg.mean(axis=1, keepdims=True)
        zsgn_avg = zsgn.mean(axis=1, keepdims=True)
        uholeg_avg = uholeg.mean(axis=1, keepdims=True)

        psg_dev = psg - psg_avg2
        zsgn_dev = zsgn - zsgn_avg
        uholeg_dev = uholeg - uholeg_avg

        group3 = np.hstack([psg_dev, zsgn_dev, uholeg_dev])

        # Group4: Argmin + Gap (4개)
        psg_argmin = psg.argmin(axis=1).reshape(-1, 1).astype(np.float32)
        psg_sorted = np.sort(psg, axis=1)
        psg_min_gap = (psg_sorted[:, 1] - psg_sorted[:, 0]).reshape(-1, 1)

        zsgn_argmin = zsgn.argmin(axis=1).reshape(-1, 1).astype(np.float32)
        zsgn_sorted = np.sort(zsgn, axis=1)
        zsgn_min_gap = (zsgn_sorted[:, 1] - zsgn_sorted[:, 0]).reshape(-1, 1)

        group4 = np.hstack([psg_argmin, psg_min_gap, zsgn_argmin, zsgn_min_gap])

        # Group5: Pressurizer-SG Coupling (2개)
        psg_mean2 = psg.mean(axis=1)
        pprz_psg_ratio = (pprz / (psg_mean2 + 1e-8)).reshape(-1, 1)
        pprz_minus_psg2 = (pprz - psg[:, 1]).reshape(-1, 1)

        group5 = np.hstack([pprz_psg_ratio, pprz_minus_psg2])

        # ── 최종 결합 ──
        X_aug = np.hstack([
            X, asym_features, coupling_features,
            phase1_asym, phase1_norm_asym,
            group1, group3, group4, group5,
        ]).astype(np.float32)

        return X_aug

    def transform(self, X, y):
        return self._compute_physics(X), y, self.feature_names

    def transform_runs(self, X_runs, y_runs):
        """런 단위로 물리 피처 계산."""
        out_X = [self._compute_physics(X_run) for X_run in X_runs]
        return out_X, y_runs, self.feature_names

    def fit_transform(self, X, y, feature_names):
        self.fit(X, y, feature_names)
        return self.transform(X, y)

    def _moving_std(self, arr):
        """이동 표준편차."""
        result = pd.Series(arr).rolling(
            window=self.moving_std_window, min_periods=2
        ).std().fillna(0.0).values.astype(np.float32)
        return result


def make_feature_method_v4(name="physics_v4", **kwargs):
    """V4 피처 엔지니어링 생성 (미분 제거)."""
    if name in ("physics_v4", "physics"):
        return FeaturePhysicsV4(**kwargs)
    raise ValueError(f"unknown feature method: {name}")
