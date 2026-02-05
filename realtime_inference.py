"""
실시간 진단 시스템 (경진대회 평가용)

요구사항:
- 5초마다 추론 결과 제출
- 1분(60초) 이내에 진단 완료 필요
- NORMAL은 1분간 진단 안 하면 성공
- 진단 확정 후 변경 불가
"""

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from practice.dataloader import load_Xy, LABELS, ID2LABEL


class RealtimeDiagnosticSystem:
    """실시간 진단 시스템"""

    def __init__(self, model_path: str, feature_method="all", window_size=None, class_mapping_path=None):
        """
        Args:
            model_path: 학습된 모델 경로
            feature_method: 피처 추출 방법
            window_size: 시계열 모델용 윈도우 크기
            class_mapping_path: 학습에 사용된 클래스 매핑 파일 경로
        """
        self.model = tf.keras.models.load_model(model_path)
        self.feature_method = feature_method
        self.window_size = window_size
        self.diagnosed_class = None
        # 3초 우승 목표: 낮은 threshold + 조기 감지 전략
        self.confidence_threshold = 0.70  # 70%로 낮춤 (빠른 진단)
        self.early_detection_threshold = 0.85  # 85% 이상이면 즉시 확정
        self.min_samples = 5  # 최소 샘플 수 (너무 적으면 오탐)
        self.start_time = None

        # 학습에 사용된 클래스 로드
        self.available_classes = None
        if class_mapping_path and Path(class_mapping_path).exists():
            self.available_classes = np.load(class_mapping_path)
            print(f"[Loaded class mapping] Available classes: {[ID2LABEL[int(c)] for c in self.available_classes]}")
        else:
            # 클래스 매핑 파일이 없으면 모델 경로에서 자동 추론
            model_dir = Path(model_path).parent
            auto_mapping_path = model_path.replace("__model.keras", "__class_mapping.npy")
            if Path(auto_mapping_path).exists():
                self.available_classes = np.load(auto_mapping_path)
                print(f"[Auto-loaded class mapping] Available: {[ID2LABEL[int(c)] for c in self.available_classes]}")
            else:
                print("[Warning] No class mapping found. Assuming all 9 classes are available.")

        # 누락된 클래스 정의 (현재 데이터 기준)
        self.missing_classes = [1, 3, 5, 6]  # LOCA_HL, LOCA_RCPCSEAL, SGTR_SG2, SGTR_SG3
        if self.available_classes is not None:
            self.missing_classes = [i for i in range(9) if i not in self.available_classes]
            if self.missing_classes:
                print(f"[Missing classes] {[ID2LABEL[c] for c in self.missing_classes]}")

    def preprocess_data(self, X, y=None, feature_names=None):
        """데이터 전처리"""
        from practice.feature_method import make_feature_method

        # feature_names가 없으면 기본 생성 (300개 피처 가정)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # 피처 엔지니어링
        feat = make_feature_method(self.feature_method)
        y_temp = y if y is not None else np.zeros(len(X), dtype=np.int64)
        X_processed, y_processed, _ = feat.fit_transform(X, y_temp, feature_names)

        # 시계열 모델용 윈도우 처리
        if self.window_size:
            from practice.dataloader import create_sliding_windows_grouped

            X_processed, y_processed = create_sliding_windows_grouped(
                X_processed, y_processed, self.window_size, group_size=10
            )

        return X_processed

    def diagnose_step(self, X_chunk, elapsed_time):
        """
        한 스텝의 진단 수행

        Args:
            X_chunk: 현재까지 수집된 데이터
            elapsed_time: 시작 후 경과 시간 (초)

        Returns:
            dict: {
                'diagnosed': bool,  # 진단 확정 여부
                'class_id': int,    # 진단 클래스 ID
                'class_name': str,  # 진단 클래스 이름
                'confidence': float,  # 확신도
                'elapsed': float    # 경과 시간
            }
        """
        # 이미 진단 확정된 경우
        if self.diagnosed_class is not None:
            return {
                "diagnosed": True,
                "class_id": self.diagnosed_class,
                "class_name": ID2LABEL[self.diagnosed_class],
                "confidence": 1.0,
                "elapsed": elapsed_time,
            }

        # 데이터 전처리
        X_processed = self.preprocess_data(X_chunk, feature_names=getattr(self, 'feature_names', None))

        # 추론
        y_prob = self.model.predict(X_processed, verbose=0)

        # 평균 확률 (시계열 모델의 경우 여러 윈도우의 평균)
        avg_prob = np.mean(y_prob, axis=0)
        predicted_class = np.argmax(avg_prob)
        confidence = float(avg_prob[predicted_class])

        # 누락된 클래스 예측 시 처리
        if predicted_class in self.missing_classes:
            # 학습되지 않은 클래스 예측 → 가장 유사한 학습된 클래스로 재매핑
            # LOCA_HL(1) → LOCA_CL(2), LOCA_RCPCSEAL(3) → LOCA_CL(2)
            # SGTR_SG2(5) → SGTR_SG1(4), SGTR_SG3(6) → SGTR_SG1(4)
            class_mapping = {
                1: 2,  # LOCA_HL → LOCA_CL
                3: 2,  # LOCA_RCPCSEAL → LOCA_CL
                5: 4,  # SGTR_SG2 → SGTR_SG1
                6: 4,  # SGTR_SG3 → SGTR_SG1
            }
            original_class = predicted_class
            predicted_class = class_mapping.get(predicted_class, predicted_class)
            confidence = float(avg_prob[predicted_class])  # 재매핑된 클래스의 확률 사용
            print(f"[Warning] Missing class {ID2LABEL[original_class]} predicted → remapped to {ID2LABEL[predicted_class]}")

        # 진단 확정 조건 (3초 우승 전략)
        diagnosed = False
        sample_count = len(X_chunk)

        # 0. 최소 샘플 수 확보 전에는 진단하지 않음
        if sample_count < self.min_samples:
            pass  # 계속 대기

        # 1. 초고확신도 (즉시 확정) - 3초 안에 맞추기 위함
        elif confidence >= self.early_detection_threshold:
            diagnosed = True
            self.diagnosed_class = predicted_class
            print(f"[Early Detection] High confidence: {confidence:.3f}")

        # 2. 3초 이내 중간 확신도 (공격적 진단)
        elif elapsed_time <= 3.0 and confidence >= 0.65:
            diagnosed = True
            self.diagnosed_class = predicted_class
            print(f"[Aggressive 3s] Confidence: {confidence:.3f}")

        # 3. 일반 확신도 (기본 전략)
        elif confidence >= self.confidence_threshold:
            diagnosed = True
            self.diagnosed_class = predicted_class

        # 4. NORMAL의 경우는 여유 있게 (1분)
        elif predicted_class == 0 and elapsed_time >= 60:
            diagnosed = True
            self.diagnosed_class = 0

        # 5. 1분 초과 (강제 종료)
        elif elapsed_time >= 60:
            diagnosed = True
            self.diagnosed_class = predicted_class

        return {
            "diagnosed": diagnosed,
            "class_id": int(predicted_class),
            "class_name": ID2LABEL[predicted_class],
            "confidence": confidence,
            "elapsed": elapsed_time,
        }

    def run_realtime_diagnosis(self, data_path: str, sampling_interval=5.0):
        """
        실시간 진단 실행

        Args:
            data_path: 실시간으로 수집되는 데이터 경로
            sampling_interval: 샘플링 간격 (초)

        Returns:
            dict: 최종 진단 결과
        """
        print(f"\n{'='*60}")
        print("실시간 진단 시스템 시작")
        print(f"모델: {self.model}")
        print(f"피처 방법: {self.feature_method}")
        print(f"샘플링 간격: {sampling_interval}초")
        print(f"{'='*60}\n")

        # 데이터 로드 (실제로는 실시간 스트리밍)
        X, y, feature_names = load_Xy(data_path, include_time=False, n_workers=1, verbose=False)
        total_samples = len(X)

        # feature_names 저장 (전처리에 사용)
        self.feature_names = feature_names

        self.start_time = time.time()
        self.diagnosed_class = None

        # 5초마다 추론
        step = 0
        chunk_size = 10  # 5초 = 10 샘플 (0.5초 간격 가정)

        while True:
            elapsed_time = time.time() - self.start_time

            # 현재까지 수집된 데이터
            end_idx = min((step + 1) * chunk_size, total_samples)
            X_chunk = X[:end_idx]

            if len(X_chunk) == 0:
                break

            # 진단 수행
            result = self.diagnose_step(X_chunk, elapsed_time)

            # 결과 출력
            print(
                f"[{elapsed_time:5.1f}s] "
                f"Class: {result['class_name']:15s} | "
                f"Confidence: {result['confidence']:.3f} | "
                f"Diagnosed: {result['diagnosed']}"
            )

            # 진단 확정 또는 1분 초과
            if result["diagnosed"]:
                print(f"\n{'='*60}")
                print(f"✓ 진단 확정: {result['class_name']}")
                print(f"  확신도: {result['confidence']:.3f}")
                print(f"  경과 시간: {result['elapsed']:.1f}초")
                print(f"{'='*60}\n")
                return result

            # 1분 초과 (실패)
            if elapsed_time >= 60:
                print(f"\n{'='*60}")
                print("✗ 진단 실패: 1분 초과")
                print(f"{'='*60}\n")
                return {
                    "diagnosed": False,
                    "class_id": -1,
                    "class_name": "TIMEOUT",
                    "confidence": 0.0,
                    "elapsed": elapsed_time,
                }

            step += 1

            # 다음 샘플링까지 대기
            time.sleep(sampling_interval)

            # 더 이상 데이터가 없으면 종료
            if end_idx >= total_samples:
                break

        # 정상 판정 (1분간 진단 안 함)
        if self.diagnosed_class == 0 or self.diagnosed_class is None:
            print(f"\n{'='*60}")
            print("✓ 정상 (NORMAL) 판정")
            print(f"{'='*60}\n")
            return {
                "diagnosed": True,
                "class_id": 0,
                "class_name": "NORMAL",
                "confidence": 1.0,
                "elapsed": 60.0,
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="실시간 진단 시스템")
    parser.add_argument("--model", type=str, required=True, help="모델 경로 (.keras)")
    parser.add_argument("--data", type=str, required=True, help="데이터 폴더")
    parser.add_argument("--feature", type=str, default="all", help="피처 방법")
    parser.add_argument("--window", type=int, default=None, help="윈도우 크기 (시계열 모델)")
    parser.add_argument("--interval", type=float, default=5.0, help="샘플링 간격 (초)")
    parser.add_argument("--class_mapping", type=str, default=None, help="클래스 매핑 파일 (.npy)")
    args = parser.parse_args()

    # 진단 시스템 초기화
    system = RealtimeDiagnosticSystem(
        model_path=args.model,
        feature_method=args.feature,
        window_size=args.window,
        class_mapping_path=args.class_mapping
    )

    # 실시간 진단 실행
    result = system.run_realtime_diagnosis(args.data, sampling_interval=args.interval)

    # 최종 결과
    print("\n최종 진단 결과:")
    print(f"  진단: {result['class_name']}")
    print(f"  확신도: {result['confidence']:.3f}")
    print(f"  소요 시간: {result['elapsed']:.1f}초")
