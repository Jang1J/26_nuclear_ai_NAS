# 변경 사항 (9개 클래스 확장 대응)

## 2026-02-05: 옵션 1 전략 구현

### 목표
5개 클래스로 먼저 완성하되, 9개 클래스로 즉시 확장 가능한 구조 구축

---

## 주요 변경사항

### 1. 클래스 매핑 자동 저장/로드 (`practice/main.py`)

**변경 위치**: main.py:294-303, 307-315, 434-447

**기능**:
- 학습 시 실제 사용된 클래스 ID를 `{model_name}__class_mapping.npy`로 저장
- 추론 시 자동으로 로드하여 학습된 클래스 확인
- 앙상블 모드도 동일하게 지원

**예시**:
```bash
# 학습 시
python -m practice.main --data_folder data/data_new --model_type mlp --train

# 저장되는 파일:
# - models/mlp__feat=all__val=0__ep=100__cw=0__seed=0__model.keras
# - models/mlp__feat=all__val=0__ep=100__cw=0__seed=0__class_mapping.npy
#   내용: [0, 2, 4, 7, 8] (현재 가용 클래스)
```

---

### 2. 실시간 추론 시스템 개선 (`realtime_inference.py`)

**변경 위치**: realtime_inference.py:18-45, 91-105, 228

**기능**:
- 클래스 매핑 파일 자동 로드
- 누락된 클래스 예측 시 유사 클래스로 자동 재매핑
- 재매핑 규칙:
  - LOCA_HL(1) → LOCA_CL(2)
  - LOCA_RCPCSEAL(3) → LOCA_CL(2)
  - SGTR_SG2(5) → SGTR_SG1(4)
  - SGTR_SG3(6) → SGTR_SG1(4)

**사용법**:
```bash
python realtime_inference.py \
  --model models/mlp__model.keras \
  --data data/data_new \
  --class_mapping models/mlp__class_mapping.npy
```

---

### 3. 시각화 개선 (`practice/utils_plot.py`)

**변경 위치**: utils_plot.py:35-98

**기능**:
- **Confusion Matrix**: 9x9 전체 표시, 누락 클래스는 회색 배경
- **Per-class Accuracy**: 누락 클래스는 회색 바로 표시, "N/A" 레이블

**변경 전**:
- 실제 존재하는 5개 클래스만 표시 (0, 2, 4, 7, 8)

**변경 후**:
- 9개 클래스 전체 표시 (0-8)
- 누락 클래스(1, 3, 5, 6) 시각적으로 구분

---

### 4. 경진대회 인터페이스 (`competition_interface.py`)

**새 파일 생성**

**기능**:
- 평가 시스템과의 데이터 송수신 인터페이스 템플릿
- 5초마다 결과 제출
- 1분 타임아웃 처리
- 표준 JSON 출력 형식

**구현 필요 항목**:
- `receive_data()`: 평가 시스템으로부터 데이터 수신 프로토콜
- `submit_result()`: 결과 제출 프로토콜

**사용법**:
```bash
python competition_interface.py \
  --model models/your_model.keras \
  --class_mapping models/your_model__class_mapping.npy \
  --timeout 60 \
  --interval 5
```

---

## 테스트 방법

### 1. 학습 테스트 (클래스 매핑 저장 확인)
```bash
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp \
  --epochs 2 \
  --train

# 확인:
ls models/*class_mapping.npy
```

### 2. 추론 테스트 (클래스 매핑 로드 확인)
```bash
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp \
  --epochs 2

# 출력에서 "[Loaded class mapping]" 확인
```

### 3. 시각화 테스트 (9개 클래스 표시 확인)
```bash
# 학습 후 test_results/ 폴더에서 확인:
# - confusion_matrix.png (9x9, 누락 클래스 회색)
# - per_class_accuracy.png (9개 바, 누락 클래스 회색)
```

---

## 데이터 확장 시나리오

### 현재 상태 (5개 클래스)
- NORMAL (0): 4,489 샘플
- LOCA_CL (2): 4,950 샘플
- SGTR_SG1 (4): 4,943 샘플
- MSLB_in (7): 4,960 샘플
- MSLB_out (8): 4,706 샘플

### 데이터 추가 시 (9개 클래스)

1. **데이터 추가**:
   ```
   data/data_new/
   ├── LOCA_HL_*.csv (새로 추가)
   ├── LOCA_RCPCSEAL_*.csv (새로 추가)
   ├── SGTR_SG2_*.csv (새로 추가)
   ├── SGTR_SG3_*.csv (새로 추가)
   └── ... (기존 파일)
   ```

2. **재학습**:
   ```bash
   python -m practice.main \
     --data_folder data/data_new \
     --model_type mlp \
     --epochs 100 \
     --train
   ```

3. **자동 확장**:
   - `class_mapping.npy`에 [0, 1, 2, 3, 4, 5, 6, 7, 8] 저장
   - 시각화 자동으로 9개 클래스 전체 활성화
   - 추론 시 재매핑 불필요

---

## 장점

✅ **즉시 사용 가능**: 5개 클래스로 지금 바로 학습/추론 가능
✅ **확장 용이**: 데이터만 추가하면 재학습만으로 9개 클래스 지원
✅ **일관된 구조**: 항상 9개 클래스 ID 사용 (0-8)
✅ **시각적 명확성**: 누락 클래스를 회색으로 명확히 표시
✅ **안전한 추론**: 학습되지 않은 클래스 예측 시 자동 재매핑

---

## 다음 단계

1. **데이터 확보**:
   - 멘토에게 LOCA_HL, LOCA_RCPCSEAL, SGTR_SG2/SG3 데이터 요청

2. **경진대회 인터페이스 완성**:
   - 평가 시스템의 데이터 송수신 프로토콜 확인
   - `competition_interface.py`의 `receive_data()`, `submit_result()` 구현

3. **하이퍼파라미터 튜닝**:
   - 9개 클래스 전체 데이터로 최적 모델 탐색
   - 앙상블 조합 최적화

4. **실전 테스트**:
   - 경진대회 형식으로 모의 평가
   - 1분 이내 진단 완료 테스트
