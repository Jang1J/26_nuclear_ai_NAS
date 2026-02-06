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

    def __init__(self, model_path: str, feature_method=None, window_size=None, class_mapping_path=None):
        from practice.preprocessing import PreprocessingPipeline

        self.model = tf.keras.models.load_model(model_path)
        self.window_size = window_size
        self.diagnosed_class = None
        self.confidence_threshold = 0.70
        self.early_detection_threshold = 0.85
        self.min_samples = 5
        self.start_time = None

        # 전처리 파이프라인 로드
        model_dir = Path(model_path).parent
        try:
            self.preprocessing = PreprocessingPipeline.load(model_dir)
            print(f"[Preprocessing pipeline loaded]")
        except Exception as e:
            print(f"[Warning] PreprocessingPipeline load failed: {e}")
            import joblib
            from practice.feature_method import make_feature_method

            scaler_path = model_path.replace("__model.keras", "__scaler.pkl")
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
            else:
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")

            if feature_method == "selection":
                selector_path = model_dir / "feature_selector_lgbm.pkl"
                if selector_path.exists():
                    self.feature_method = make_feature_method("selection", model_path=str(selector_path), save_model=False)
                else:
                    raise FileNotFoundError(f"Selector not found: {selector_path}")
            else:
                self.feature_method = make_feature_method(feature_method or "all")

            self.preprocessing = None

        # 클래스 매핑 로드
        self.available_classes = None
        if class_mapping_path and Path(class_mapping_path).exists():
            self.available_classes = np.load(class_mapping_path)
        else:
            auto_mapping_path = model_path.replace("__model.keras", "__class_mapping.npy")
            if Path(auto_mapping_path).exists():
                self.available_classes = np.load(auto_mapping_path)

        n_classes = len(LABELS)
        self.missing_classes = []
        if self.available_classes is not None:
            self.missing_classes = [i for i in range(n_classes) if i not in self.available_classes]
            if self.missing_classes:
                print(f"[Missing classes] {[ID2LABEL[c] for c in self.missing_classes]}")

    def preprocess_data(self, X, y=None, feature_names=None):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if self.preprocessing is not None:
            X_processed = self.preprocessing.transform(X, y, feature_names)
        else:
            y_temp = y if y is not None else np.zeros(len(X), dtype=np.int64)
            X_processed, _, _ = self.feature_method.transform(X, y_temp)
            X_processed = self.scaler.transform(X_processed)

        # 시계열 모델: 슬라이딩 윈도우 생성
        if self.window_size and len(X_processed) >= self.window_size:
            N, D = X_processed.shape
            windows = []
            for i in range(N - self.window_size + 1):
                windows.append(X_processed[i : i + self.window_size])
            X_processed = np.array(windows, dtype=np.float32)

        return X_processed

    def diagnose_step(self, X_chunk, elapsed_time):
        if self.diagnosed_class is not None:
            return {
                "diagnosed": True,
                "class_id": self.diagnosed_class,
                "class_name": ID2LABEL[self.diagnosed_class],
                "confidence": 1.0,
                "elapsed": elapsed_time,
            }

        X_processed = self.preprocess_data(X_chunk, feature_names=getattr(self, 'feature_names', None))

        y_prob = self.model.predict(X_processed, verbose=0)
        avg_prob = np.mean(y_prob, axis=0)

        if self.missing_classes:
            avg_prob[self.missing_classes] = 0.0

        prob_sum = np.sum(avg_prob)
        if prob_sum > 0:
            avg_prob = avg_prob / prob_sum

        predicted_class = np.argmax(avg_prob)
        confidence = float(avg_prob[predicted_class])

        diagnosed = False
        sample_count = len(X_chunk)

        if sample_count < self.min_samples:
            pass
        elif confidence >= self.early_detection_threshold:
            diagnosed = True
            self.diagnosed_class = predicted_class
        elif elapsed_time <= 3.0 and confidence >= 0.65:
            diagnosed = True
            self.diagnosed_class = predicted_class
        elif confidence >= self.confidence_threshold:
            diagnosed = True
            self.diagnosed_class = predicted_class
        elif predicted_class == 0 and elapsed_time >= 60:
            diagnosed = True
            self.diagnosed_class = 0
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
        print(f"\n{'='*60}")
        print("realtime diagnosis start")
        print(f"{'='*60}\n")

        X, y, feature_names = load_Xy(data_path, include_time=False, n_workers=1, verbose=False)
        total_samples = len(X)
        self.feature_names = feature_names

        self.start_time = time.time()
        self.diagnosed_class = None

        step = 0
        chunk_size = 5  # 5초 = 5 samples (1초 간격)

        while True:
            elapsed_time = time.time() - self.start_time

            end_idx = min((step + 1) * chunk_size, total_samples)
            X_chunk = X[:end_idx]

            if len(X_chunk) == 0:
                break

            result = self.diagnose_step(X_chunk, elapsed_time)

            print(
                f"[{elapsed_time:5.1f}s] "
                f"Class: {result['class_name']:15s} | "
                f"Confidence: {result['confidence']:.3f} | "
                f"Diagnosed: {result['diagnosed']}"
            )

            if result["diagnosed"]:
                print(f"\nDiagnosed: {result['class_name']} (conf={result['confidence']:.3f}, time={result['elapsed']:.1f}s)")
                return result

            if elapsed_time >= 60:
                return {
                    "diagnosed": False,
                    "class_id": -1,
                    "class_name": "TIMEOUT",
                    "confidence": 0.0,
                    "elapsed": elapsed_time,
                }

            step += 1
            time.sleep(sampling_interval)

            if end_idx >= total_samples:
                break

        return {
            "diagnosed": True,
            "class_id": 0,
            "class_name": "NORMAL",
            "confidence": 1.0,
            "elapsed": 60.0,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--feature", type=str, default="all")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--class_mapping", type=str, default=None)
    args = parser.parse_args()

    system = RealtimeDiagnosticSystem(
        model_path=args.model,
        feature_method=args.feature,
        window_size=args.window,
        class_mapping_path=args.class_mapping
    )

    result = system.run_realtime_diagnosis(args.data, sampling_interval=args.interval)
    print(f"\nFinal: {result['class_name']} (conf={result['confidence']:.3f})")
