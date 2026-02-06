"""
경진대회 실시간 평가 인터페이스

대회 흐름:
  - NORMAL 데이터가 계속 오다가 특정 시점에 사고가 발생
  - 사고 발생 후 1분 안에 진단해야 함
  - 매 1초마다 멘토 PC → 팀 PC: CNS 데이터 1행
  - 매 1초마다 팀 PC → 멘토 PC: 결과 dict 반환
    {
      'results': 'NORMAL' or 'ESDE_in' 등,  # 현재 판단 클래스명
      'Diagnostic_time': None or float,       # 사고 진단 확정 시간
      'Class probabilities': [0.98, ...]      # 9개 클래스 확률
    }
  - 끊지 않고 60초 끝까지 매초 결과 반환
  - 10회 시나리오 반복
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

from practice.dataloader import LABELS, ID2LABEL, LABEL2ID, _resolve_useless_json_path
from practice.preprocessing import PreprocessingPipeline


def is_complete_model(model_path: str) -> bool:
    """
    모델 아티팩트 3종(model.keras, feature_transformer.pkl, preprocessing_metadata.json)이
    모두 존재하는지 확인.
    """
    prefix = model_path.replace("__model.keras", "")
    required = [
        f"{prefix}__feature_transformer.pkl",
        f"{prefix}__preprocessing_metadata.json",
    ]
    return all(Path(r).exists() for r in required)


def find_valid_models(models_dir: str = "models") -> list:
    """
    models/ 디렉토리에서 유효한 모델 후보만 반환.
    - ._* (macOS 리소스 포크) 제외
    - __MACOSX 경로 제외
    - 아티팩트 3종 완전한 모델만 포함
    """
    import glob as _glob

    # 프로젝트 루트 기준으로도 탐색
    models_path = Path(models_dir)
    if not models_path.is_dir():
        project_root = Path(__file__).resolve().parent
        models_path = project_root / models_dir
    if not models_path.is_dir():
        return []

    candidates = sorted(_glob.glob(str(models_path / "*__model.keras")))

    valid = []
    for c in candidates:
        basename = Path(c).name
        # macOS 리소스 포크 제외
        if basename.startswith("._"):
            continue
        # __MACOSX 경로 제외
        if "__MACOSX" in c:
            continue
        # 아티팩트 완전성 체크
        if not is_complete_model(c):
            print(f"[WARNING] Incomplete model (missing pipeline artifacts), skipping: {basename}")
            continue
        valid.append(c)

    return valid


class CompetitionSystem:
    """
    경진대회 실시간 진단 시스템.

    매 1초마다 1행 수신 → 즉시 결과 dict 반환.
    NORMAL 상태에서 사고 감지 시 전환.
    """

    def __init__(self, model_path: str, window_size=3,
                 useless_features_json: str = "useless_features_all.json"):
        # 커스텀 객체 등록 (WarmupCosineSchedule 등)
        try:
            from practice.main import WarmupCosineSchedule  # noqa: F401
        except Exception:
            pass

        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception:
            self.model = tf.keras.models.load_model(model_path, compile=False)

        self.window_size = window_size

        # 모델 경로에서 prefix 자동 추출 (Goal A: 파이프라인 격리)
        model_dir = Path(model_path).parent
        model_filename = Path(model_path).stem  # "xxx__model" → prefix = "xxx"
        prefix = model_filename.replace("__model", "") if "__model" in model_filename else None

        # prefix 기반으로 올바른 파이프라인 로드
        self.preprocessing = PreprocessingPipeline.load(model_dir, prefix=prefix)

        # useless features 목록 로드 (CWD 독립 경로 탐색)
        self._useless_features = set()
        resolved_useless = _resolve_useless_json_path(useless_features_json)
        if Path(resolved_useless).exists():
            with open(resolved_useless, 'r') as f:
                useless_data = json.load(f)
            self._useless_features = set(useless_data.get("useless_features", []))

        # 원본→파이프라인 피처 매핑 (set_feature_names 호출 시 설정)
        self._keep_indices = None

        # 클래스 매핑 로드
        class_mapping_path = model_path.replace("__model.keras", "__class_mapping.npy")
        if Path(class_mapping_path).exists():
            self.available_classes = np.load(class_mapping_path)
            self.missing_classes = [i for i in range(len(LABELS)) if i not in self.available_classes]
        else:
            self.available_classes = None
            self.missing_classes = []

        self.reset()

    def set_feature_names(self, all_feature_names):
        """
        원본 데이터의 전체 컬럼명을 설정.
        학습 시 사용된 피처와 매칭하여 인덱스 + 순서 보장.

        Goal C 강화:
        - 순서 보장: 파이프라인이 기대하는 피처 순서대로 재배열
        - 누락 감지: 학습에 사용된 피처가 입력에 없으면 경고
        - 추가 피처 무시: 입력에만 있는 피처는 자동 제거

        Args:
            all_feature_names: list — 원본 CSV의 전체 컬럼명 (KCNTOMS 제외 후)
        """
        expected_names = self.preprocessing.feature_names_in
        input_name_to_idx = {name: i for i, name in enumerate(all_feature_names)}

        # 파이프라인이 기대하는 순서대로 인덱스 매핑
        self._keep_indices = []
        self._reorder_indices = []
        missing_features = []

        for expected_name in expected_names:
            if expected_name in input_name_to_idx:
                self._keep_indices.append(input_name_to_idx[expected_name])
            else:
                missing_features.append(expected_name)

        n_removed = len(all_feature_names) - len(self._keep_indices)
        n_matched = len(self._keep_indices)

        print(f"[CompetitionSystem] Feature mapping: "
              f"{len(all_feature_names)} input → {n_matched} matched "
              f"({n_removed} removed)")

        if missing_features:
            print(f"  ⚠️ WARNING: {len(missing_features)} expected features missing!")
            print(f"     Missing: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
            print(f"     Model may produce degraded predictions.")

        if n_matched == 0:
            raise ValueError(
                "No matching features found! Check if the data source matches training data."
            )

    def reset(self):
        """새 시나리오 시작 시 호출."""
        self.buffer = []
        self.step = 0
        self.diagnosed_time = None    # 사고 진단 확정 시간
        self.diagnosed_class = None   # 확정된 사고 클래스명
        self._last_probs = [0.0] * len(LABELS)

        # Goal D: 연속 합의 기반 확정 로직
        self._consecutive_class = None  # 연속으로 예측된 클래스
        self._consecutive_count = 0     # 연속 횟수
        self._prob_history = []         # 최근 확률 이력 (마진 계산용)

    def receive_and_predict(self, row_data):
        """
        매 1초마다 호출.

        Args:
            row_data: np.ndarray (D,) — CNS 센서 1행

        Returns:
            dict: {
                'results': str,                    # 현재 판단 ('NORMAL' or 사고 클래스명)
                'Diagnostic_time': None or float,  # 사고 진단 확정 시점
                'Class probabilities': list        # 9개 클래스 확률
            }
        """
        self.step += 1
        elapsed = float(self.step)

        row = row_data.astype(np.float32)

        # useless features 제거
        if self._keep_indices is not None:
            row = row[self._keep_indices]

        # 버퍼에 추가
        self.buffer.append(row)
        X_chunk = np.array(self.buffer)

        # 전처리
        X_processed = self.preprocessing.transform(X_chunk)

        # 윈도우 부족 → NORMAL 반환
        if len(X_processed) < self.window_size:
            self._last_probs = [0.0] * len(LABELS)
            return {
                'results': 'NORMAL',
                'Diagnostic_time': None,
                'Class probabilities': self._last_probs,
            }

        # 최근 윈도우 1개로 추론
        X_window = X_processed[-self.window_size:]
        X_input = X_window.reshape(1, self.window_size, -1)

        y_prob = self.model.predict(X_input, verbose=0)[0]

        # missing class 마스킹
        if self.missing_classes:
            y_prob[self.missing_classes] = 0.0
            prob_sum = np.sum(y_prob)
            if prob_sum > 0:
                y_prob = y_prob / prob_sum

        probs = y_prob.tolist()
        self._last_probs = probs

        predicted_class = int(np.argmax(y_prob))
        confidence = float(y_prob[predicted_class])
        predicted_name = ID2LABEL[predicted_class]

        # 이미 사고 확정된 경우 → 계속 같은 결과
        if self.diagnosed_time is not None:
            return {
                'results': self.diagnosed_class,
                'Diagnostic_time': self.diagnosed_time,
                'Class probabilities': probs,
            }

        # NORMAL이면 연속 카운트 리셋 후 NORMAL 반환
        if predicted_class == LABEL2ID["NORMAL"]:
            self._consecutive_class = None
            self._consecutive_count = 0
            self._prob_history = []
            return {
                'results': 'NORMAL',
                'Diagnostic_time': None,
                'Class probabilities': probs,
            }

        # ────────────────────────────────────────────
        # Goal D: 연속 합의 + 마진 기반 확정 로직
        # ────────────────────────────────────────────
        # 1등-2등 확률 차이 (마진)
        sorted_probs = sorted(y_prob, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

        # 연속 카운트 갱신
        if predicted_class == self._consecutive_class:
            self._consecutive_count += 1
        else:
            self._consecutive_class = predicted_class
            self._consecutive_count = 1
            self._prob_history = []

        self._prob_history.append(confidence)

        # 확정 조건 (3단계 트리거)
        diagnosed = False

        # Tier 1: 매우 높은 확신 + 연속 2회 → 즉시 확정 (빠른 진단)
        if confidence >= 0.90 and self._consecutive_count >= 2 and margin >= 0.3:
            diagnosed = True

        # Tier 2: 높은 확신 + 연속 3회 → 확정 (안정적 진단)
        elif confidence >= 0.70 and self._consecutive_count >= 3 and margin >= 0.15:
            diagnosed = True

        # Tier 3: 보통 확신 + 연속 5회 → 확정 (보수적 진단)
        elif confidence >= 0.50 and self._consecutive_count >= 5:
            # 최근 5회 평균 확신도도 체크
            avg_conf = np.mean(self._prob_history[-5:])
            if avg_conf >= 0.55:
                diagnosed = True

        # Tier 4: 시간 초과 대비 (55초 이상 경과 시 최선의 판단)
        elif elapsed >= 55.0 and confidence >= 0.40:
            diagnosed = True

        if diagnosed:
            self.diagnosed_class = predicted_name
            self.diagnosed_time = elapsed
            return {
                'results': predicted_name,
                'Diagnostic_time': elapsed,
                'Class probabilities': probs,
            }
        else:
            # 사고 감지했지만 아직 확정 불충분 → 클래스명은 보여줌
            return {
                'results': predicted_name,
                'Diagnostic_time': None,
                'Class probabilities': probs,
            }


# ─────────────────────────────────────────────
# 로컬 테스트용
# ─────────────────────────────────────────────
def local_test(model_path, data_csv_path, window_size=3):
    """
    단일 CSV 파일로 대회 시뮬레이션.
    60초 끝까지 매초 결과 출력 (break 없음).
    """
    import pandas as pd

    system = CompetitionSystem(model_path=model_path, window_size=window_size)

    df = pd.read_csv(data_csv_path)
    if "KCNTOMS" in df.columns:
        df = df.drop(columns=["KCNTOMS"])

    # 원본 컬럼명으로 useless features 매핑 설정
    system.set_feature_names(list(df.columns))

    print(f"\n{'='*70}")
    print(f"Local test: {data_csv_path}")
    print(f"Samples: {len(df)}, Features: {len(df.columns)}")
    print(f"{'='*70}\n")

    for i in range(min(len(df), 120)):  # 최대 120초 (NORMAL + 사고 60초)
        row = df.iloc[i].values
        result = system.receive_and_predict(row)

        probs = result['Class probabilities']
        top_prob = max(probs) if any(p > 0 for p in probs) else 0
        diag = f"@ {result['Diagnostic_time']}s" if result['Diagnostic_time'] else ""

        print(f"[{i+1:3d}s] {result['results']:15s} | conf={top_prob:.3f} {diag}")

    print(f"\n{'='*70}")
    print(f"Final: results={result['results']}, Diagnostic_time={result['Diagnostic_time']}")
    print(f"{'='*70}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="single CSV file")
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    local_test(args.model, args.data, args.window)
