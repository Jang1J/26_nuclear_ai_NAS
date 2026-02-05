"""
경진대회 평가 인터페이스

요구사항:
- 평가 시스템으로부터 데이터 수신
- 5초마다 진단 결과 제출
- 1분 이내 진단 완료
- 표준 출력 형식으로 결과 반환
"""

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from practice.dataloader import LABELS, ID2LABEL
from realtime_inference import RealtimeDiagnosticSystem


class CompetitionInterface:
    """경진대회 평가 인터페이스"""

    def __init__(self, model_path: str, feature_method="all", window_size=None, class_mapping_path=None):
        """
        Args:
            model_path: 학습된 모델 경로
            feature_method: 피처 추출 방법
            window_size: 시계열 모델용 윈도우 크기
            class_mapping_path: 클래스 매핑 파일 경로
        """
        self.diagnostic_system = RealtimeDiagnosticSystem(
            model_path=model_path,
            feature_method=feature_method,
            window_size=window_size,
            class_mapping_path=class_mapping_path
        )
        self.submission_count = 0

    def receive_data(self):
        """
        평가 시스템으로부터 데이터 수신 (구현 필요)

        TODO: 실제 경진대회에서 제공하는 데이터 수신 프로토콜에 맞춰 구현

        Returns:
            numpy.ndarray: 수신된 센서 데이터 (N, D)
        """
        # 예시: 소켓 통신, REST API, 파일 기반 등
        # data = socket.recv()
        # return np.array(data)
        raise NotImplementedError("데이터 수신 프로토콜을 구현해야 합니다.")

    def submit_result(self, diagnosis_result: dict):
        """
        평가 시스템으로 결과 제출

        Args:
            diagnosis_result: {
                'diagnosed': bool,
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'elapsed': float
            }

        TODO: 실제 경진대회에서 요구하는 출력 형식에 맞춰 구현
        """
        self.submission_count += 1

        # 표준 출력 형식 (예시)
        output = {
            "submission_id": self.submission_count,
            "timestamp": time.time(),
            "diagnosed": diagnosis_result['diagnosed'],
            "class_id": diagnosis_result['class_id'],
            "class_name": diagnosis_result['class_name'],
            "confidence": diagnosis_result['confidence'],
            "elapsed_time": diagnosis_result['elapsed']
        }

        # JSON 형식으로 출력 (예시)
        import json
        print(json.dumps(output, ensure_ascii=False))

        # 또는 소켓/REST API로 전송
        # socket.send(json.dumps(output))

        return output

    def run_competition_mode(self, timeout=60.0, sampling_interval=5.0):
        """
        경진대회 모드 실행

        Args:
            timeout: 최대 진단 시간 (초)
            sampling_interval: 샘플링 간격 (초)

        Returns:
            dict: 최종 진단 결과
        """
        print(f"\n{'='*60}")
        print("경진대회 모드 시작")
        print(f"제한 시간: {timeout}초")
        print(f"샘플링 간격: {sampling_interval}초")
        print(f"{'='*60}\n")

        start_time = time.time()
        data_buffer = []  # 누적 데이터

        while True:
            elapsed = time.time() - start_time

            # 1. 데이터 수신
            try:
                new_data = self.receive_data()
                data_buffer.append(new_data)
            except NotImplementedError:
                print("[Warning] 데이터 수신 프로토콜이 구현되지 않았습니다.")
                print("테스트를 위해 로컬 데이터를 사용하세요.")
                return None

            # 2. 진단 수행
            X_chunk = np.vstack(data_buffer)
            result = self.diagnostic_system.diagnose_step(X_chunk, elapsed)

            # 3. 결과 제출 (5초마다)
            if elapsed % sampling_interval < 0.1:  # 오차 허용
                self.submit_result(result)

            # 4. 종료 조건
            if result['diagnosed']:
                print(f"\n✓ 진단 확정: {result['class_name']}")
                print(f"  확신도: {result['confidence']:.3f}")
                print(f"  소요 시간: {result['elapsed']:.1f}초")
                return result

            if elapsed >= timeout:
                print(f"\n✗ 타임아웃: {timeout}초 초과")
                # NORMAL 판정 또는 실패 처리
                if result['class_id'] == 0:
                    print("✓ NORMAL 판정")
                    result['diagnosed'] = True
                else:
                    print("✗ 진단 실패")
                    result['diagnosed'] = False
                return result

            # 다음 샘플링까지 대기
            time.sleep(sampling_interval)


def example_usage():
    """사용 예시"""
    # 모델 경로 설정
    model_path = "models/mlp_v2__feat=all__val=1__ep=100__cw=1__seed=0__model.keras"
    class_mapping_path = "models/mlp_v2__feat=all__val=1__ep=100__cw=1__seed=0__class_mapping.npy"

    # 인터페이스 초기화
    interface = CompetitionInterface(
        model_path=model_path,
        feature_method="all",
        window_size=None,
        class_mapping_path=class_mapping_path
    )

    # 경진대회 모드 실행
    result = interface.run_competition_mode(timeout=60.0, sampling_interval=5.0)

    if result:
        print("\n최종 결과:")
        print(f"  진단: {result['class_name']}")
        print(f"  확신도: {result['confidence']:.3f}")
        print(f"  소요 시간: {result['elapsed']:.1f}초")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="경진대회 평가 인터페이스")
    parser.add_argument("--model", type=str, required=True, help="모델 경로 (.keras)")
    parser.add_argument("--feature", type=str, default="all", help="피처 방법")
    parser.add_argument("--window", type=int, default=None, help="윈도우 크기")
    parser.add_argument("--class_mapping", type=str, default=None, help="클래스 매핑 파일")
    parser.add_argument("--timeout", type=float, default=60.0, help="제한 시간 (초)")
    parser.add_argument("--interval", type=float, default=5.0, help="샘플링 간격 (초)")
    args = parser.parse_args()

    interface = CompetitionInterface(
        model_path=args.model,
        feature_method=args.feature,
        window_size=args.window,
        class_mapping_path=args.class_mapping
    )

    result = interface.run_competition_mode(timeout=args.timeout, sampling_interval=args.interval)
