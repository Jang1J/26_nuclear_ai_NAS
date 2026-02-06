"""
시계열 데이터 증강 모듈.

원전 사고 센서 데이터에 특화된 증강 기법:
1. Jitter: 가우시안 노이즈 추가 (센서 노이즈 시뮬레이션)
2. Scaling: 랜덤 진폭 스케일링 (센서 캘리브레이션 편차 시뮬레이션)
3. Mixup: 같은 클래스 내 선형 보간 (다양성 증가)
4. MinorityOversampler: 소수 클래스 타겟 오버샘플링
"""
import numpy as np
from typing import Tuple, Optional, List


class TimeSeriesAugmenter:
    """
    시계열 데이터 증강.
    물리적 제약을 존중하면서 다양성 증가.
    """

    def __init__(
        self,
        jitter_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        mixup_alpha: float = 0.0,
        augment_prob: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            jitter_std: 가우시안 노이즈 표준편차 (신호 std 대비 비율)
            scale_range: (min, max) 스케일링 범위
            mixup_alpha: Mixup beta 분포 파라미터 (0 = 비활성)
            augment_prob: 증강 적용 확률
            seed: 랜덤 시드
        """
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.mixup_alpha = mixup_alpha
        self.augment_prob = augment_prob
        self.rng = np.random.default_rng(seed)

    def jitter(self, X: np.ndarray) -> np.ndarray:
        """
        가우시안 노이즈 추가 (센서 노이즈 시뮬레이션).

        Args:
            X: (N, W, D) or (N, D)

        Returns:
            X_aug: 동일 shape
        """
        # 피처별 표준편차 계산 → 그에 비례하는 노이즈
        if X.ndim == 3:
            std = np.std(X, axis=(0, 1), keepdims=True)  # (1, 1, D)
        else:
            std = np.std(X, axis=0, keepdims=True)  # (1, D)

        std = np.maximum(std, 1e-8)  # 0 방지
        noise = self.rng.normal(0, self.jitter_std * std, X.shape)
        return (X + noise).astype(np.float32)

    def scaling(self, X: np.ndarray) -> np.ndarray:
        """
        랜덤 진폭 스케일링 (센서 캘리브레이션 편차 시뮬레이션).

        Args:
            X: (N, W, D) or (N, D)

        Returns:
            X_aug: 동일 shape
        """
        # 피처별 독립적 스케일링
        n_features = X.shape[-1]
        scale = self.rng.uniform(
            self.scale_range[0], self.scale_range[1], size=(n_features,)
        ).astype(np.float32)
        return X * scale

    def mixup(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        같은 클래스 내에서만 Mixup (안전성 보장).

        Args:
            X: (N, W, D) or (N, D)
            y: (N,)

        Returns:
            X_aug, y_aug: 원본 + 증강 데이터
        """
        if self.mixup_alpha <= 0:
            return X, y

        X_new, y_new = [], []

        for label in np.unique(y):
            mask = y == label
            X_cls = X[mask]
            n = len(X_cls)
            if n < 2:
                continue

            n_mix = n // 2
            lam = self.rng.beta(self.mixup_alpha, self.mixup_alpha, size=n_mix)

            idx = self.rng.permutation(n)
            for i in range(n_mix):
                i1, i2 = idx[2 * i], idx[2 * i + 1]
                l = lam[i]
                if X.ndim == 3:
                    l = l.reshape(1, 1)
                else:
                    l = l.reshape(1)
                x_mix = (l * X_cls[i1] + (1 - l) * X_cls[i2]).astype(np.float32)
                X_new.append(x_mix)
                y_new.append(label)

        if len(X_new) > 0:
            X_mixed = np.stack(X_new)
            y_mixed = np.array(y_new, dtype=np.int64)
            X = np.concatenate([X, X_mixed], axis=0)
            y = np.concatenate([y, y_mixed])

        return X, y

    def augment(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        원본 데이터에 증강 데이터를 추가하여 반환.

        원본은 그대로 유지, 증강된 복사본을 추가.

        Args:
            X: (N, W, D) or (N, D)
            y: (N,)

        Returns:
            X_aug, y_aug: 원본 + 증강 데이터
        """
        # 증강 대상 샘플 선택
        n = len(X)
        aug_mask = self.rng.random(n) < self.augment_prob
        n_aug = aug_mask.sum()

        if n_aug == 0:
            return X, y

        X_to_aug = X[aug_mask].copy()
        y_to_aug = y[aug_mask].copy()

        # Jitter + Scaling 적용
        X_augmented = self.jitter(X_to_aug)
        X_augmented = self.scaling(X_augmented)

        # 원본 + 증강 합침
        X_out = np.concatenate([X, X_augmented], axis=0)
        y_out = np.concatenate([y, y_to_aug])

        # Mixup (옵션)
        if self.mixup_alpha > 0:
            X_out, y_out = self.mixup(X_out, y_out)

        # 셔플
        perm = self.rng.permutation(len(X_out))
        return X_out[perm], y_out[perm]


class MinorityOversampler:
    """
    소수 클래스 타겟 오버샘플링.
    ESDE_out 등 성능이 낮은 클래스에 대해 집중적으로 증강.
    """

    def __init__(
        self,
        target_classes: List[int],
        target_ratio: float = 2.0,
        jitter_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.90, 1.10),
        seed: Optional[int] = None,
    ):
        """
        Args:
            target_classes: 증강할 클래스 ID 목록 (예: [8] = ESDE_out)
            target_ratio: 증강 비율 (2.0 = 원본의 2배로)
            jitter_std: 더 강한 노이즈 (소수 클래스 다양성 증가)
            scale_range: 더 넓은 스케일링 범위
            seed: 랜덤 시드
        """
        self.target_classes = target_classes
        self.target_ratio = target_ratio
        self.augmenter = TimeSeriesAugmenter(
            jitter_std=jitter_std,
            scale_range=scale_range,
            augment_prob=1.0,
            seed=seed,
        )

    def oversample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        소수 클래스만 증강하여 추가.

        Args:
            X: (N, W, D) or (N, D)
            y: (N,)

        Returns:
            X_aug, y_aug: 원본 + 소수 클래스 증강
        """
        X_parts = [X]
        y_parts = [y]

        for cls_id in self.target_classes:
            mask = y == cls_id
            X_cls = X[mask]
            y_cls = y[mask]
            n_orig = len(X_cls)

            if n_orig == 0:
                continue

            n_aug = int(n_orig * (self.target_ratio - 1))
            if n_aug <= 0:
                continue

            # 복원 추출 → 증강
            idx = self.augmenter.rng.choice(n_orig, size=n_aug, replace=True)
            X_sampled = X_cls[idx].copy()

            X_augmented = self.augmenter.jitter(X_sampled)
            X_augmented = self.augmenter.scaling(X_augmented)

            X_parts.append(X_augmented)
            y_parts.append(np.full(n_aug, cls_id, dtype=np.int64))

        X_out = np.concatenate(X_parts, axis=0)
        y_out = np.concatenate(y_parts)

        # 셔플
        perm = self.augmenter.rng.permutation(len(X_out))
        return X_out[perm], y_out[perm]
