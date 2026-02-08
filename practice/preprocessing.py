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

    def __init__(self, feature_method="physics", **feature_kwargs):
        """
        Args:
            feature_method: 피처 엔지니어링 방법 (physics)
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

        # 입력 피처 검증 (Goal C: 유연한 검증 + 명확한 경고)
        if feature_names is not None:
            input_names = list(feature_names)
            if input_names != self.feature_names_in:
                # 수량만 다른지, 이름이 다른지 구분
                expected_set = set(self.feature_names_in)
                input_set = set(input_names)
                missing = expected_set - input_set
                extra = input_set - expected_set

                if missing:
                    import warnings
                    warnings.warn(
                        f"Feature names mismatch: {len(missing)} expected features missing.\n"
                        f"Missing: {sorted(missing)[:5]}...\n"
                        f"Expected {len(self.feature_names_in)}, got {len(input_names)}.\n"
                        f"Proceeding with available features (may degrade accuracy)."
                    )
                elif extra and not missing:
                    # 추가 피처만 있음 (이미 제거된 상태라면 OK)
                    pass
                else:
                    raise ValueError(
                        f"Feature names mismatch!\n"
                        f"Expected: {self.feature_names_in[:5]}...\n"
                        f"Got: {input_names[:5]}..."
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

    def save(self, directory, prefix=None):
        """
        전처리 파이프라인 저장

        Args:
            directory: 저장 디렉토리 (예: "models/")
            prefix: 파일명 prefix (예: "cnn_attention__feat=physics__...")
                    None이면 기존처럼 generic 이름 사용 (하위 호환성)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline")

        # prefix가 있으면 모델별 격리 파일명, 없으면 기존 generic 이름
        if prefix:
            scaler_name = f"{prefix}__scaler.pkl"
            feat_name = f"{prefix}__feature_transformer.pkl"
            meta_name = f"{prefix}__preprocessing_metadata.json"
        else:
            scaler_name = "scaler.pkl"
            feat_name = "feature_transformer.pkl"
            meta_name = "preprocessing_metadata.json"

        # 1. Scaler 저장
        scaler_path = directory / scaler_name
        joblib.dump(self.scaler, scaler_path)

        # 2. Feature transformer 저장 (selection의 경우 중요)
        feat_transformer_path = directory / feat_name
        joblib.dump(self.feature_transformer, feat_transformer_path)

        # 3. 메타데이터 저장
        metadata = {
            "feature_method": self.feature_method_name,
            "feature_kwargs": self.feature_kwargs,
            "feature_names_in": self.feature_names_in,
            "feature_names_out": self.feature_names_out,
        }
        metadata_path = directory / meta_name
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[Saved preprocessing pipeline]")
        print(f"  - Scaler: {scaler_path}")
        print(f"  - Feature transformer: {feat_transformer_path}")
        print(f"  - Metadata: {metadata_path}")

    @classmethod
    def load(cls, directory, prefix=None):
        """
        전처리 파이프라인 로드

        Args:
            directory: 로드 디렉토리 (예: "models/")
            prefix: 파일명 prefix (예: "cnn_attention__feat=physics__...")
                    None이면 자동 탐색: prefix 파일 → generic 파일 순서로 시도

        Returns:
            pipeline: PreprocessingPipeline 객체
        """
        directory = Path(directory)

        # prefix 결정 로직
        if prefix:
            metadata_path = directory / f"{prefix}__preprocessing_metadata.json"
            scaler_path = directory / f"{prefix}__scaler.pkl"
            feat_transformer_path = directory / f"{prefix}__feature_transformer.pkl"

            # prefix 파일이 없으면 generic으로 fallback (v1 모델 호환)
            if not metadata_path.exists():
                generic_meta = directory / "preprocessing_metadata.json"
                if generic_meta.exists():
                    print(f"[Pipeline] Prefix '{prefix}' files not found, "
                          f"falling back to generic pipeline")
                    metadata_path = generic_meta
                    scaler_path = directory / "scaler.pkl"
                    feat_transformer_path = directory / "feature_transformer.pkl"
        else:
            # 자동 탐색: prefix 파일이 있으면 사용, 없으면 generic
            metadata_path = directory / "preprocessing_metadata.json"
            scaler_path = directory / "scaler.pkl"
            feat_transformer_path = directory / "feature_transformer.pkl"

            # generic 파일이 없으면 prefix 파일 자동 탐색
            if not metadata_path.exists():
                candidates = sorted(directory.glob("*__preprocessing_metadata.json"))
                if candidates:
                    # 가장 최근 파일 사용
                    meta_file = candidates[-1]
                    found_prefix = meta_file.name.replace("__preprocessing_metadata.json", "")
                    metadata_path = meta_file
                    scaler_path = directory / f"{found_prefix}__scaler.pkl"
                    feat_transformer_path = directory / f"{found_prefix}__feature_transformer.pkl"
                    print(f"[Auto-detected pipeline prefix] {found_prefix}")

        # 1. 메타데이터 로드
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # 2. 파이프라인 생성
        pipeline = cls(
            feature_method=metadata["feature_method"],
            **metadata["feature_kwargs"]
        )

        # 3. Scaler 로드
        pipeline.scaler = joblib.load(scaler_path)

        # 4. Feature transformer 로드
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
