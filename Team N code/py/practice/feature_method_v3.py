"""
Physics 기반 피처 엔지니어링 v3 (Phase 2).

v2(FeaturePhysics)를 상속하여 Phase 2 피처를 추가.
목적: SGTR leak=1에서 Loop 간 구분력 강화.

Phase 2 신규 피처 (~30개):
  - Group 1: Pairwise Differences (9개) — 루프 간 직접 비교
  - Group 2: Derivative Differences (6개) — 변화율 차이
  - Group 3: Deviation from Average (9개) — 평균 대비 편차
  - Group 4: Argmin + Gap (4개) — 비정상 루프 식별 + 구분 명확도
  - Group 5: Pressurizer-SG Coupling (2개) — Loop2 가압기 정규화
"""
import numpy as np
from .feature_method import FeaturePhysics


class FeaturePhysicsV3(FeaturePhysics):
    """
    v2 + Phase 2 피처.
    총 피처: 252 (v2) + 30 (Phase 2) = 282개.
    """

    def fit(self, X, y, feature_names):
        # v2 fit 호출 (인덱스 매핑 + v2 피처 이름 생성)
        super().fit(X, y, feature_names)

        # Phase 2 추가 인덱스 (v2에서 이미 매핑된 것 재사용)
        # self._sg_p_idx → PSG1, PSG2, PSG3
        # self._sgn_idx → ZSGN1, ZSGN2, ZSGN3
        # self._hotleg_idx → UHOLEG1, UHOLEG2, UHOLEG3
        # self._idx_PPRZ → PPRZ

        # Phase 2 피처 이름 정의
        self._phase2_names = []

        # Group 1: Pairwise Differences (9개)
        self._phase2_names += [
            "PSG1-PSG2", "PSG2-PSG3", "PSG1-PSG3",
            "ZSGN1-ZSGN2", "ZSGN2-ZSGN3", "ZSGN1-ZSGN3",
            "UHOLEG1-UHOLEG2", "UHOLEG2-UHOLEG3", "UHOLEG1-UHOLEG3",
        ]

        # Group 2: Derivative Differences (6개)
        self._phase2_names += [
            "dPSG1-dPSG2", "dPSG2-dPSG3", "dPSG1-dPSG3",
            "dZSGN1-dZSGN2", "dZSGN2-dZSGN3", "dZSGN1-dZSGN3",
        ]

        # Group 3: Deviation from Average (9개)
        self._phase2_names += [
            "PSG1-PSGavg", "PSG2-PSGavg", "PSG3-PSGavg",
            "ZSGN1-ZSGNavg", "ZSGN2-ZSGNavg", "ZSGN3-ZSGNavg",
            "UHOLEG1-UHOLEGavg", "UHOLEG2-UHOLEGavg", "UHOLEG3-UHOLEGavg",
        ]

        # Group 4: Argmin + Gap (4개)
        self._phase2_names += [
            "PSG_argmin", "PSG_min_gap",
            "ZSGN_argmin", "ZSGN_min_gap",
        ]

        # Group 5: Pressurizer-SG Coupling (2개)
        self._phase2_names += [
            "PPRZ_PSGavg_ratio", "PPRZ-PSG2",
        ]

        # 전체 피처 이름 갱신 (v2 + Phase 2)
        self.feature_names = self.feature_names + self._phase2_names

        return self

    def _compute_physics(self, X):
        """v2 피처 + Phase 2 피처 계산."""
        # 1) v2 피처 계산 (base + 48 physics)
        X_v2 = super()._compute_physics(X)

        N = X.shape[0]

        # ──────────────────────────────────────────
        # 원본 데이터에서 필요한 변수 추출
        # ──────────────────────────────────────────
        psg = X[:, self._sg_p_idx]       # (N, 3) PSG1, PSG2, PSG3
        zsgn = X[:, self._sgn_idx]       # (N, 3) ZSGN1, ZSGN2, ZSGN3
        uholeg = X[:, self._hotleg_idx]  # (N, 3) UHOLEG1, UHOLEG2, UHOLEG3
        pprz = X[:, self._idx_PPRZ]      # (N,)

        # 1차 미분
        diff1 = np.zeros_like(X)
        diff1[1:] = X[1:] - X[:-1]

        d_psg = diff1[:, self._sg_p_idx]   # (N, 3)
        d_zsgn = diff1[:, self._sgn_idx]   # (N, 3)

        # ──────────────────────────────────────────
        # Group 1: Pairwise Differences (9개)
        # ──────────────────────────────────────────
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

        # ──────────────────────────────────────────
        # Group 2: Derivative Differences (6개)
        # ──────────────────────────────────────────
        d_psg_diff_12 = (d_psg[:, 0] - d_psg[:, 1]).reshape(-1, 1)
        d_psg_diff_23 = (d_psg[:, 1] - d_psg[:, 2]).reshape(-1, 1)
        d_psg_diff_13 = (d_psg[:, 0] - d_psg[:, 2]).reshape(-1, 1)

        d_zsgn_diff_12 = (d_zsgn[:, 0] - d_zsgn[:, 1]).reshape(-1, 1)
        d_zsgn_diff_23 = (d_zsgn[:, 1] - d_zsgn[:, 2]).reshape(-1, 1)
        d_zsgn_diff_13 = (d_zsgn[:, 0] - d_zsgn[:, 2]).reshape(-1, 1)

        group2 = np.hstack([
            d_psg_diff_12, d_psg_diff_23, d_psg_diff_13,
            d_zsgn_diff_12, d_zsgn_diff_23, d_zsgn_diff_13,
        ])

        # ──────────────────────────────────────────
        # Group 3: Deviation from Average (9개)
        # ──────────────────────────────────────────
        psg_avg = psg.mean(axis=1, keepdims=True)
        zsgn_avg = zsgn.mean(axis=1, keepdims=True)
        uholeg_avg = uholeg.mean(axis=1, keepdims=True)

        psg_dev = psg - psg_avg       # (N, 3)
        zsgn_dev = zsgn - zsgn_avg    # (N, 3)
        uholeg_dev = uholeg - uholeg_avg  # (N, 3)

        group3 = np.hstack([psg_dev, zsgn_dev, uholeg_dev])  # (N, 9)

        # ──────────────────────────────────────────
        # Group 4: Argmin + Gap (4개)
        # ──────────────────────────────────────────
        # PSG: 가장 낮은 압력 = 누설 발생 루프
        psg_argmin = psg.argmin(axis=1).reshape(-1, 1).astype(np.float32)
        psg_sorted = np.sort(psg, axis=1)
        psg_min_gap = (psg_sorted[:, 1] - psg_sorted[:, 0]).reshape(-1, 1)

        # ZSGN: 가장 낮은 수위 = 누설 발생 루프
        zsgn_argmin = zsgn.argmin(axis=1).reshape(-1, 1).astype(np.float32)
        zsgn_sorted = np.sort(zsgn, axis=1)
        zsgn_min_gap = (zsgn_sorted[:, 1] - zsgn_sorted[:, 0]).reshape(-1, 1)

        group4 = np.hstack([psg_argmin, psg_min_gap, zsgn_argmin, zsgn_min_gap])

        # ──────────────────────────────────────────
        # Group 5: Pressurizer-SG Coupling (2개)
        # ──────────────────────────────────────────
        psg_mean = psg.mean(axis=1)
        pprz_psg_ratio = (pprz / (psg_mean + 1e-8)).reshape(-1, 1)
        pprz_minus_psg2 = (pprz - psg[:, 1]).reshape(-1, 1)

        group5 = np.hstack([pprz_psg_ratio, pprz_minus_psg2])

        # ──────────────────────────────────────────
        # 최종: v2 + Phase 2
        # ──────────────────────────────────────────
        X_v3 = np.hstack([
            X_v2, group1, group2, group3, group4, group5,
        ]).astype(np.float32)

        return X_v3


def make_feature_method_v3(name="physics_v3", **kwargs):
    """V3 피처 엔지니어링 생성."""
    if name in ("physics_v3", "physics"):
        return FeaturePhysicsV3(**kwargs)
    raise ValueError(f"unknown feature method: {name}")
