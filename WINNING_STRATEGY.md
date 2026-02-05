# 🏆 3초 진단 우승 전략

## 목표
**3초 안에 정확하게 사고 유형 진단** → 최고 점수

---

## 📊 현재 상황 분석

### 시간별 데이터량 (추정)
- 센서 샘플링: 0.5초 간격 (추정)
- 3초 = **6 샘플**
- 5초 = **10 샘플**
- 10초 = **20 샘플**

### 사고 특성
원자력 사고는 **초반에 급격한 변화** 발생:

| 사고 | 초반 신호 (0-3초) | 주요 센서 |
|------|------------------|---------|
| LOCA | 가압기 압력/레벨 급락 | PPRZ↓↓, ZPRZ↓↓ |
| SGTR | 해당 SG 압력 상승 | PSG1/2/3 ↑, 방사능↑ |
| MSLB | 증기압 급락 | PSG ↓↓, 온도↓ |
| NORMAL | 변화 없음 | 모든 센서 안정 |

---

## 🎯 우승 전략

### 1. **Feature Engineering: 변화율 강조** ⭐⭐⭐

**diff 피처 사용 (이미 구현됨)**:
```bash
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp_v2 \
  --feature_method diff \
  --epochs 100 \
  --use_val \
  --use_class_weight \
  --train
```

**physics 피처 사용 (최고 추천)**:
```bash
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn1d \
  --feature_method physics \
  --window_size 10 \
  --epochs 100 \
  --use_val \
  --use_class_weight \
  --train
```

**장점**:
- 1차/2차 미분으로 **급격한 변화 감지**
- 루프 간 비대칭으로 **SGTR/MSLB 파손 위치 식별**
- 3초 내 적은 샘플로도 높은 정확도

---

### 2. **모델 선택: 시계열 + 앙상블** ⭐⭐⭐

**추천 조합**:
1. **CNN1D** (급격한 변화 포착)
2. **LSTM** (시간 흐름 학습)
3. **MLP_v2** (전체 패턴 학습)

**앙상블 학습**:
```bash
python -m practice.main \
  --data_folder data/data_new \
  --ensemble \
  --ensemble_method weighted_vote \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --use_val \
  --use_class_weight \
  --train
```

**window_size=6**: 3초 데이터만으로도 예측 가능하도록 설정

---

### 3. **추론 전략: 공격적 조기 진단** ⭐⭐⭐

**realtime_inference.py에서 구현 완료**:

```python
# 0. 최소 샘플 확보 (5개)
min_samples = 5

# 1. 초고확신도 (85% 이상) → 즉시 확정
early_detection_threshold = 0.85

# 2. 3초 이내 중간 확신도 (65%) → 공격적 진단
if elapsed_time <= 3.0 and confidence >= 0.65:
    diagnosed = True

# 3. 일반 확신도 (70%)
confidence_threshold = 0.70
```

**전략**:
- **3초 이내**: 확신도 65%만 되면 진단
- **3초 이후**: 70% 이상 필요
- **매우 확실**: 85% 이상이면 즉시 확정

---

### 4. **데이터 증강: 초반 샘플 집중** ⭐⭐

**학습 시 초반 데이터 강조**:
- 사고 발생 후 0-10초 데이터 가중치 증가
- 초반 급격한 변화 패턴 학습 강화

**방법**:
```python
# data_split.py에서 구현 가능
# 그룹 내 초반 샘플에 높은 가중치
sample_weights = np.ones(len(y))
sample_weights[::group_size] *= 2.0  # 각 그룹 첫 샘플 2배
```

---

### 5. **하이퍼파라미터 최적화** ⭐⭐

**목표**: 3초 데이터로 최고 정확도

| 파라미터 | 추천값 | 이유 |
|---------|--------|------|
| window_size | 6-10 | 3-5초 데이터 커버 |
| batch_size | 64-128 | 빠른 학습 |
| learning_rate | 1e-3 | 기본값 유지 |
| dropout | 0.2-0.3 | 과적합 방지 (적은 샘플) |
| use_class_weight | True | 클래스 불균형 처리 |

---

## 📋 실행 계획

### Phase 1: 최적 모델 탐색 (현재)

```bash
# 1. Diff 피처 + MLP
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp_v2 \
  --feature_method diff \
  --epochs 100 \
  --use_val \
  --use_class_weight \
  --train

# 2. Physics 피처 + CNN1D
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn1d \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --use_val \
  --use_class_weight \
  --train

# 3. 앙상블
python -m practice.main \
  --data_folder data/data_new \
  --ensemble \
  --ensemble_method weighted_vote \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --use_val \
  --use_class_weight \
  --train
```

### Phase 2: 실시간 테스트

```bash
# 3초 진단 테스트
python realtime_inference.py \
  --model models/best_model.keras \
  --data data/data_new \
  --feature physics \
  --window 6 \
  --interval 1.0 \
  --class_mapping models/best_model__class_mapping.npy
```

**목표**: 3초 이내 80%+ 정확도

### Phase 3: 9개 클래스 데이터 추가 시

1. 누락된 데이터 추가
2. 전체 재학습
3. 경진대회 제출

---

## 🔬 성능 평가 기준

| 진단 시간 | 점수 가중치 | 전략 |
|----------|-----------|------|
| 0-3초 | 최고 (100%) | **목표 구간** |
| 3-10초 | 높음 (80%) | 허용 범위 |
| 10-30초 | 중간 (50%) | 재학습 필요 |
| 30-60초 | 낮음 (20%) | 실패 수준 |
| 60초+ | 0% | 타임아웃 |

---

## ⚡ 핵심 성공 요소

### 1. **변화율 피처** (가장 중요!)
- diff: 1차 변화율
- physics: 1차/2차 미분
- → 급격한 변화를 즉시 포착

### 2. **적은 샘플로 높은 정확도**
- window_size=6 (3초)
- 초반 데이터 강조 학습

### 3. **공격적 threshold**
- 3초 이내: 65% 확신도만으로 진단
- 초고확신: 85% 이상 즉시 확정

### 4. **앙상블 신뢰도**
- 3개 모델 결합
- 개별 모델 약점 보완

---

## 🎓 추가 개선 아이디어

### 1. 사고별 맞춤 전략
```python
# LOCA: 압력 급락 패턴 → 0.5초면 충분
if PPRZ_drop_rate > threshold:
    return LOCA

# SGTR: SG 비대칭 → 2초 필요
if PSG_asymmetry > threshold:
    return SGTR

# MSLB: 증기압 급락 → 1초면 충분
if PSG_drop_rate > threshold:
    return MSLB
```

### 2. 룰 베이스 + AI 하이브리드
```python
# 물리적 룰 먼저 체크 (0.1초)
if rule_check(X) is not None:
    return rule_result

# 룰 불충분 → AI 예측 (3초)
else:
    return model.predict(X)
```

### 3. 조기 종료 최적화
```python
# 연속 3번 같은 클래스 예측 → 즉시 확정
if last_3_predictions == [LOCA, LOCA, LOCA]:
    return LOCA
```

---

## 📌 체크리스트

- [x] 변화율 피처 (diff/physics) 구현됨
- [x] 공격적 threshold 적용
- [x] 3초 조기 진단 로직 구현
- [ ] window_size=6으로 모델 학습
- [ ] 9개 클래스 전체 데이터 확보
- [ ] 실시간 테스트 (3초 진단율 측정)
- [ ] 앙상블 모델 최적화
- [ ] 경진대회 인터페이스 완성

---

## 🚀 다음 단계

1. **즉시 실행**: physics 피처 + CNN1D + window_size=6 학습
2. **성능 측정**: 3초 시점 정확도 확인
3. **threshold 튜닝**: 65% → 60%? 70%?
4. **앙상블 최적화**: weighted_vote vs stacking

**목표: 3초 안에 85%+ 정확도 달성!** 🏆
