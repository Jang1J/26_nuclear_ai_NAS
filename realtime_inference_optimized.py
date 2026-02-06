"""
실시간 진단 시스템 - 속도 최적화 버전 (GPT 제안 2번)

핵심 개선:
1. 전처리는 전체 데이터에 대해 1회만
2. 매 제출 시점마다 "최근 window 1개"만 잘라서 예측
3. 샘플 index 기반 가상시간으로 제출 타이밍 계산
"""

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from practice.dataloader import LABELS, ID2LABEL
from practice.preprocessing import PreprocessingPipeline


class RealtimeDiagnosticSystemOptimized:
    """
    속도 최적화된 실시간 진단 시스템

    GPT 제안 적용:
    - 전처리 1회 → 속도 50% 이상 향상
    - 샘플 index 기반 타이밍 → 정확한 제출 간격
    """

    def __init__(self, model_path: str, window_size=3):
        """
        Args:
            model_path: 학습된 모델 경로
            window_size: 시계열 윈도우 크기 (3=3초 @1초 샘플링)
        """
        self.model = tf.keras.models.load_model(model_path)
        self.window_size = window_size
        self.diagnosed_class = None

        # 3초 우승 전략
        self.confidence_threshold = 0.70
        self.early_detection_threshold = 0.85
        self.min_samples = 5

        # 전처리 파이프라인 로드
        model_dir = Path(model_path).parent
        self.preprocessing = PreprocessingPipeline.load(model_dir)

        # 클래스 매핑 로드
        class_mapping_path = model_path.replace("__model.keras", "__class_mapping.npy")
        if Path(class_mapping_path).exists():
            self.available_classes = np.load(class_mapping_path)
            self.missing_classes = [i for i in range(len(LABELS)) if i not in self.available_classes]
        else:
            self.available_classes = None
            self.missing_classes = []

        print(f"✅ [Optimized System Loaded]")
        print(f"   - Model: {Path(model_path).name}")
        print(f"   - Window: {window_size}")
        print(f"   - Available classes: {len(self.available_classes) if self.available_classes is not None else 9}")

    def diagnose_stream(
        self,
        X_stream,
        feature_names,
        sampling_interval=1.0,
        submit_interval=5.0,
        max_time=60.0,
    ):
        """
        전체 스트림에 대한 진단 수행 (최적화 버전)

        Args:
            X_stream: (N, D) 전체 데이터 스트림
            feature_names: D개 피처 이름
            sampling_interval: 샘플링 간격 (초)
            submit_interval: 제출 간격 (초)
            max_time: 최대 진단 시간 (초)

        Returns:
            results: 제출 시점별 진단 결과 리스트
        """
        print(f"\n{'='*70}")
        print(f"🔬 속도 최적화 실시간 진단 시작")
        print(f"{'='*70}")
        print(f"스트림 길이: {len(X_stream)} samples")
        print(f"샘플링 간격: {sampling_interval}s")
        print(f"제출 간격: {submit_interval}s")
        print(f"최대 시간: {max_time}s")

        # ✅ GPT 제안 2번: 전처리는 1회만!
        print(f"\n[1/3] 전처리 (1회 실행)...")
        start_time = time.time()
        X_processed = self.preprocessing.transform(X_stream, feature_names=feature_names)
        preprocess_time = time.time() - start_time
        print(f"   ✓ 전처리 완료: {X_processed.shape} ({preprocess_time:.2f}s)")

        # 제출 타이밍 계산
        samples_per_submit = int(submit_interval / sampling_interval)
        max_samples = int(max_time / sampling_interval)

        results = []
        self.diagnosed_class = None

        print(f"\n[2/3] 실시간 진단 시작...")
        print(f"   제출 간격: {samples_per_submit} samples ({submit_interval}s)")

        for current_idx in range(self.window_size, min(len(X_processed), max_samples), samples_per_submit):
            elapsed_time = current_idx * sampling_interval

            # 진단 수행
            result = self._diagnose_at_index(X_processed, current_idx, elapsed_time)
            results.append(result)

            # 로그 출력
            status = "✅ 확정" if result["diagnosed"] else "⏳ 대기"
            print(f"   [{elapsed_time:5.1f}s] {status} | {result['class_name']:15s} ({result['confidence']:.3f})")

            # 진단 확정되면 종료
            if result["diagnosed"]:
                print(f"\n✅ [진단 확정] {elapsed_time:.1f}초에 {result['class_name']} 진단!")
                break

        print(f"\n[3/3] 진단 완료")
        print(f"{'='*70}")

        return results

    def _diagnose_at_index(self, X_processed, current_idx, elapsed_time):
        """특정 index에서의 진단 수행"""
        # 이미 확정된 경우
        if self.diagnosed_class is not None:
            return {
                "diagnosed": True,
                "class_id": self.diagnosed_class,
                "class_name": ID2LABEL[self.diagnosed_class],
                "confidence": 1.0,
                "elapsed": elapsed_time,
            }

        # ✅ GPT 제안 2번: 최근 window 1개만 잘라서 예측
        start_idx = max(0, current_idx - self.window_size)
        X_window = X_processed[start_idx:current_idx]

        # 윈도우가 부족하면 패딩
        if len(X_window) < self.window_size:
            pad_size = self.window_size - len(X_window)
            X_window = np.vstack([np.zeros((pad_size, X_window.shape[1])), X_window])

        # (1, window, features) 형태로
        X_input = X_window.reshape(1, self.window_size, -1)

        # 추론
        y_prob = self.model.predict(X_input, verbose=0)
        avg_prob = y_prob[0]  # (num_classes,)

        # Missing class 마스킹
        if self.missing_classes:
            avg_prob[self.missing_classes] = 0.0
            prob_sum = np.sum(avg_prob)
            if prob_sum > 0:
                avg_prob = avg_prob / prob_sum

        predicted_class = np.argmax(avg_prob)
        confidence = float(avg_prob[predicted_class])

        # 진단 확정 조건
        diagnosed = False
        sample_count = current_idx

        if sample_count < self.min_samples:
            pass
        elif confidence >= self.early_detection_threshold:
            diagnosed = True
            self.diagnosed_class = predicted_class
        elif confidence >= self.confidence_threshold and sample_count >= 20:
            diagnosed = True
            self.diagnosed_class = predicted_class
        elif elapsed_time >= 55.0 and predicted_class == 0:
            diagnosed = True
            self.diagnosed_class = 0

        return {
            "diagnosed": diagnosed,
            "class_id": predicted_class,
            "class_name": ID2LABEL[predicted_class],
            "confidence": confidence,
            "elapsed": elapsed_time,
        }


# 사용 예시
if __name__ == "__main__":
    from practice.dataloader import load_Xy

    # 데이터 로드
    X, y, feature_names = load_Xy("data/data_new", include_time=False, verbose=False)

    # LOCA 사고 데이터 1개 선택
    loca_indices = np.where(y == 2)[0]
    start_idx = loca_indices[0]
    X_stream = X[start_idx : start_idx + 600]  # 60초 (1Hz 가정)

    # 진단 시스템 생성
    system = RealtimeDiagnosticSystemOptimized(
        model_path="models/cnn_attention_physics__model.keras",
        window_size=3
    )

    # 진단 수행
    results = system.diagnose_stream(
        X_stream,
        feature_names,
        sampling_interval=1.0,
        submit_interval=5.0,
        max_time=60.0
    )
