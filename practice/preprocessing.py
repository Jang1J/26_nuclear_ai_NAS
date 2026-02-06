"""
전처리 파이프라인 통합 클래스 (GPT 제안 1번)

학습과 추론이 100% 동일한 전처리를 사용하도록 보장
"""
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from .feature_method import make_feature_method


class PreprocessingPipeline:
    """
    학습=추론 100% 일치 전처리 파이프라인

    Usage (학습):
        pipeline = PreprocessingPipeline(feature_method="physics")
        X_processed = pipeline.fit_transform(X, y, feature_names)
        pipeline.save("models/exp1/")

    Usage (추론):
        pipeline = PreprocessingPipeline.load("models/exp1/")
        X_processed = pipeline.transform(X, feature_names)
    """

    def __init__(self, feature_method="all", **feature_kwargs):
        """
        Args:
            feature_method: all, diff, stats, physics, selection
            **feature_kwargs: feature_method별 추가 인자
        """
        self.feature_method_name = feature_method
        self.feature_kwargs = feature_kwargs

        # fit 시 생성될 객체들
        self.feature_transformer = None
        self.scaler = None
        self.feature_names_in = None
        self.feature_names_out = None
        self._is_fitted = False

    def fit(self, X, y, feature_names):
        """
        전처리 파이프라인 학습

        Args:
            X: (N, D) 원본 데이터
            y: (N,) 라벨
            feature_names: D개 피처 이름

        Returns:
            self
        """
        self.feature_names_in = list(feature_names)

        # 1. Feature engineering fit
        self.feature_transformer = make_feature_method(
            self.feature_method_name,
            **self.feature_kwargs
        )
        X_feat, _, self.feature_names_out = self.feature_transformer.fit_transform(
            X, y, feature_names
        )

        # 2. Scaler fit
        self.scaler = StandardScaler()
        self.scaler.fit(X_feat)

        self._is_fitted = True
        return self

    def transform(self, X, y=None, feature_names=None):
        """
        전처리 파이프라인 적용 (transform-only)

        Args:
            X: (N, D) 원본 데이터
            y: (N,) 라벨 (선택, feature_transformer가 필요로 할 수 있음)
            feature_names: D개 피처 이름 (검증용)

        Returns:
            X_processed: (N, D') 전처리된 데이터
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() or load() first.")

        # 입력 피처 검증
        if feature_names is not None:
            if list(feature_names) != self.feature_names_in:
                raise ValueError(
                    f"Feature names mismatch!\n"
                    f"Expected: {self.feature_names_in[:5]}...\n"
                    f"Got: {list(feature_names)[:5]}..."
                )

        # y가 없으면 더미 생성 (일부 feature_transformer가 y를 요구)
        if y is None:
            y = np.zeros(len(X), dtype=np.int64)

        # 1. Feature engineering transform
        X_feat, _, _ = self.feature_transformer.transform(X, y)

        # 2. Scaler transform
        X_scaled = self.scaler.transform(X_feat)

        return X_scaled

    def fit_transform(self, X, y, feature_names):
        """
        fit + transform 동시 수행

        Returns:
            X_processed: (N, D') 전처리된 데이터
        """
        self.fit(X, y, feature_names)
        return self.transform(X, y, feature_names)

    def save(self, directory):
        """
        전처리 파이프라인 저장

        Args:
            directory: 저장 디렉토리 (예: "models/exp1/")
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline")

        # 1. Scaler 저장
        scaler_path = directory / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        # 2. Feature transformer 저장 (selection의 경우 중요)
        feat_transformer_path = directory / "feature_transformer.pkl"
        joblib.dump(self.feature_transformer, feat_transformer_path)

        # 3. 메타데이터 저장
        metadata = {
            "feature_method": self.feature_method_name,
            "feature_kwargs": self.feature_kwargs,
            "feature_names_in": self.feature_names_in,
            "feature_names_out": self.feature_names_out,
        }
        metadata_path = directory / "preprocessing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[Saved preprocessing pipeline]")
        print(f"  - Scaler: {scaler_path}")
        print(f"  - Feature transformer: {feat_transformer_path}")
        print(f"  - Metadata: {metadata_path}")

    @classmethod
    def load(cls, directory):
        """
        전처리 파이프라인 로드

        Args:
            directory: 로드 디렉토리 (예: "models/exp1/")

        Returns:
            pipeline: PreprocessingPipeline 객체
        """
        directory = Path(directory)

        # 1. 메타데이터 로드
        metadata_path = directory / "preprocessing_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 2. 파이프라인 생성
        pipeline = cls(
            feature_method=metadata["feature_method"],
            **metadata["feature_kwargs"]
        )

        # 3. Scaler 로드
        scaler_path = directory / "scaler.pkl"
        pipeline.scaler = joblib.load(scaler_path)

        # 4. Feature transformer 로드
        feat_transformer_path = directory / "feature_transformer.pkl"
        pipeline.feature_transformer = joblib.load(feat_transformer_path)

        # 5. 메타데이터 복원
        pipeline.feature_names_in = metadata["feature_names_in"]
        pipeline.feature_names_out = metadata["feature_names_out"]
        pipeline._is_fitted = True

        print(f"[Loaded preprocessing pipeline]")
        print(f"  - Feature method: {pipeline.feature_method_name}")
        print(f"  - Input features: {len(pipeline.feature_names_in)}")
        print(f"  - Output features: {len(pipeline.feature_names_out)}")

        return pipeline

    def get_feature_names_out(self):
        """출력 피처 이름 반환"""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")
        return self.feature_names_out
