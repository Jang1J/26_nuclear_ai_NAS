# GPT 브리핑: 원전 사고 진단 모델 프로젝트 현황

## 1. 프로젝트 개요

**목적**: 원자력 발전소 시계열 센서 데이터를 실시간으로 분석하여 9가지 사고 유형을 분류하는 딥러닝 모델 개발

**대회 형식**:
- 서버에서 1초마다 CSV 파일이 생성됨 (test{id}_sec{n}.csv)
- 각 테스트는 60초간 진행 (sec1 ~ sec60)
- 모델이 실시간으로 매초 추론하고, **비정상이 연속 3초 동일하면 사고 확정** → UDP로 결과 전송
- 빠르고 정확하게 사고를 확정하는 것이 핵심 (확정 후 번복 불가)

**9개 클래스**:
| ID | 클래스명 | 설명 |
|----|----------|------|
| 0 | NORMAL | 정상 운전 |
| 1 | LOCA_HL | Hot Leg 냉각재 상실 사고 |
| 2 | LOCA_CL | Cold Leg 냉각재 상실 사고 |
| 3 | LOCA_RCP | RCP Seal 냉각재 상실 사고 |
| 4 | SGTR_Loop1 | 증기발생기 세관파단 루프1 |
| 5 | SGTR_Loop2 | 증기발생기 세관파단 루프2 |
| 6 | SGTR_Loop3 | 증기발생기 세관파단 루프3 |
| 7 | ESDE_in | 격납건물 내부 증기배관 파단 |
| 8 | ESDE_out | 격납건물 외부 증기배관 파단 |

---

## 2. 데이터 구조

### 2-1. 학습 데이터 (`data/data_new/`)
- **1,759개 CSV 파일**, 각 파일 = 1개 시뮬레이션 런
- 각 CSV: 310행 × 301컬럼 (300개 센서 피처 + KCNTOMS 시간 컬럼)
- 0.5초 간격 샘플링 → 310행 = 155초
- **클래스별 파일 수 (심각한 불균형)**:
  - NORMAL: 4개 (각 310행 = 1,240 샘플)
  - LOCA_HL / LOCA_CL / LOCA_RCP: 각 450개 (각 ~139,500 샘플)
  - SGTR_Loop1 / Loop2 / Loop3: 각 15개 (각 ~4,650 샘플)
  - ESDE_in / ESDE_out: 각 180개 (각 ~55,800 샘플)

**학습 데이터의 중요한 특성**:
- 각 사고 파일의 **앞 ~10행은 사실상 NORMAL 구간** (delay period). 사고가 발생하기 전 정상 운전 데이터인데, 파일 전체에 사고 라벨이 붙어있음 → **오라벨 문제**
- 300개 원본 피처 중 113개는 거의 상수값 (near-constant) → 제거 → **188개 유효 피처**

### 2-2. 실제 테스트 데이터 (`data/real_test_data/`)
- **100개 테스트 시나리오** (test1 ~ test100)
- 각 시나리오 = 60개 CSV 파일 (test{id}_sec1.csv ~ test{id}_sec60.csv)
- 각 CSV: 1행 데이터 × **2,205컬럼** (학습 데이터의 300컬럼은 이 중 일부)
- **원본은 310행 데이터를 5초 간격으로 서브샘플링해서 60개 파일로 만든 것**
  - sec1 = 원본 row 1, sec2 = 원본 row 6, sec3 = 원본 row 11, ...
  - secN = 원본 row (1 + (N-1) × 5)
- `answers.csv`에 정답 존재 (label, malf_delay 등)

### 2-3. 데이터 매핑 (핵심!)

**`malf_delay`**: 원본 데이터에서 사고가 시작되는 시점 (초)

예시: test1은 SGTR_Loop1이고 malf_delay=38
- 원본 데이터에서 38번째 행(초)부터 사고 시작
- 서브샘플링된 데이터에서는: sec = (38-1)/5 + 1 ≈ **sec 8.4**
- 즉 **sec1~sec8은 완전히 NORMAL**, sec9부터 사고 신호가 나타나기 시작

**malf_delay 분포**: 0~55초, 평균 ~24초
- malf_delay=0은 NORMAL (사고 없음)
- malf_delay=55이면 서브샘플링 후 sec12까지 NORMAL → 초반에 사고 판별 불가

**테스트 클래스 분포** (100개 중):
- NORMAL: 8개, LOCA_HL: 6, LOCA_CL: 13, LOCA_RCP: 13
- SGTR_Loop1: 16, SGTR_Loop2: 11, SGTR_Loop3: 8
- ESDE_in: 12, ESDE_out: 13

---

## 3. 모델 아키텍처

### 3-1. 핵심 모델: TCN (Temporal Convolutional Network)
- **입력**: (batch, window=3, 266 features) — 3타임스텝 슬라이딩 윈도우
- **구조**: 2개 Residual Block (dilated causal conv, dilation=1,2) → GlobalAvgPool → Dense(128) → Softmax(9)
- **수용영역**: 7 타임스텝 (kernel=2, dilation 1,2)

### 3-2. 피처 엔지니어링 파이프라인

**원본 → useless 제거 → physics 피처 추가 → 스케일링**

1. **원본 300피처** → 113개 useless 제거 → **188개 유효 피처**
   - useless: near-constant 피처, 학습에 무의미한 변수
   - 특히 17개 near-constant 피처(std < 0.001)는 StandardScaler를 폭발시키는 원인

2. **Physics Feature Engineering (V3)**: 188 → **266 피처**
   - **Phase 1 (v2)**: 48개 물리 기반 파생 피처
     - 1차/2차 미분 (16개): dPPRZ, dPSG1~3, dPCTMT, dZSGNOR1~3 등
     - 루프 간 비대칭 (7개): PSG_range, ZSGNOR_range, PSG_std 등
     - 물리적 상관관계 (5개): PPRZ-PSGavg, MovStd(PPRZ) 등
     - 추가 비대칭 (17개): UHOLEG, ZSGN, WSTM 그룹
     - 핵심 미분 (3개): dVSUMP, d(PSG_range)/dt 등
   - **Phase 2 (v3)**: 30개 추가 (SGTR Loop 구분력 강화용)
     - Pairwise Differences (9개): PSG1-PSG2, ZSGN1-ZSGN2 등
     - Derivative Differences (6개): dPSG1-dPSG2 등
     - Deviation from Average (9개): PSG1-PSGavg 등
     - Argmin + Gap (4개): 어느 SG가 가장 비정상인지
     - Pressurizer-SG Coupling (2개): PPRZ/PSGavg ratio 등

3. **StandardScaler** + `np.clip(X, -10, 10)` 클리핑

### 3-3. 학습 설정
```
- Optimizer: AdamW (weight_decay=1e-4)
- Loss: Focal Loss (gamma=2.0) — 클래스 불균형 대응
- LR Schedule: Warmup Cosine (warmup 5% → cosine decay)
- LR: 1e-3
- Batch Size: 128
- Early Stopping: patience=25, val_loss 기준
- SGTR F1 Callback: 매 epoch SGTR macro-F1 계산, best 모델 저장
- Sample Weight: Leak=1 SGTR 샘플에 4배 가중치
- Class Weight: 불균형 보정 (sample_weight에 통합)
- Data Augmentation: jitter + scaling + minority oversampling (클래스 4,5,6,7,8)
- skip_delay_rows=10: 사고 파일 앞 10행 제거 (delay 오라벨 방지)
- Loop2 후처리: argmax=Loop2이고 confidence<0.55이면 2등 클래스로 교체
```

---

## 4. 실시간 추론 파이프라인 (대회 제출용)

### 4-1. 추론 흐름 (매 초)
```
CSV 읽기 (2205 컬럼)
→ 학습용 188 컬럼 추출
→ 누적 버퍼에 추가
→ Physics V3 피처 변환 (버퍼 전체에 대해, 미분 계산 위해)
→ StandardScaler + np.clip(-10, 10)
→ 마지막 3타임스텝 윈도우 추출
→ TCN 모델 추론 → 9개 클래스 확률
→ 확정 로직 적용
```

### 4-2. 진단 확정 로직 (RealtimeInference 클래스)
```python
WINDOW = 3           # 슬라이딩 윈도우 크기
CONFIRM_COUNT = 3    # 연속 N초 같은 비정상이면 확정
GRACE_PERIOD = 5     # 처음 5초는 확정 안 함 (패딩 불안정 방지)

# 확정 조건:
# 1) 아직 미확정 (confirmed == False)
# 2) 현재 예측 ≠ NORMAL
# 3) 현재 sec > GRACE_PERIOD (5)
# 4) 최근 CONFIRM_COUNT(3)개 예측이 모두 동일
# → 조건 충족 시 confirmed = True, 이후 번복 불가
```

### 4-3. 핵심 문제: 학습 데이터 vs 실제 데이터 간 도메인 갭

**학습 데이터**:
- 피처값이 소수점 이하 많은 자릿수 (float64 정밀도)
- 예: `15578262.123456789`

**실제 테스트 데이터**:
- 반올림된 값 (float32 또는 정수 수준)
- 예: `15578262.0`

**문제**:
- near-constant 피처(std ~ 1e-8)에 대해 StandardScaler가 (x - mean) / std를 계산할 때
- 학습 시 std ≈ 0.000001이면 정상 범위
- 테스트 시 반올림 차이 때문에 scaled 값이 339,661 같은 극단값으로 폭발
- → **`np.clip(X, -10, 10)`으로 해결**하고, 원인 피처 17개는 학습에서도 제거

---

## 5. 현재까지의 결과

### 5-1. 내부 테스트 (학습 데이터의 20% 홀드아웃)

**5에폭 TCN (skip_delay_rows=10, 113개 useless 제거)**:
```
전체 정확도: 99.58%
모든 클래스 F1 ≥ 0.99
SGTR_Loop1: F1=0.99, SGTR_Loop2: F1=1.00, SGTR_Loop3: F1=0.99
학습시간: 410초, 추론: 0.047ms/샘플
```

### 5-2. 실제 테스트 데이터 (100개 시나리오)

**5에폭 모델 결과: 약 12% 정확도 (매우 나쁨)**

**주요 문제: ESDE_in 편향**
- 100개 중 ~80개 이상을 ESDE_in으로 예측
- 특히 NORMAL 구간(사고 전)의 데이터를 ESDE_in으로 분류
- sec6(GRACE_PERIOD=5 직후)에서 ESDE_in으로 잘못 확정 → 이후 올바른 예측이 나와도 이미 확정됨

**test1 상세 (SGTR_Loop1, malf_delay=38)**:
```
sec1~5:  ESDE_in (~60-70% 확률) ← GRACE_PERIOD 내라 확정 안됨
sec6:    ESDE_in 확정됨 ← 문제! 아직 NORMAL 구간임
sec7~37: ESDE_in 계속 (이미 확정 상태)
sec38:   사고 시작 (PPRZ 급락)
sec39~60: SGTR_Loop1 99%+ 확률 ← 이때야 올바른 예측이지만 이미 확정 후
```

### 5-3. NORMAL 원본 데이터 테스트
- 이전 모델(skip_delay 미적용): SGTR_Loop3 73% → **오답**
- 현재 모델(skip_delay=10, 피처 정리): NORMAL 93% → **정답** ✅

---

## 6. 해결해야 할 핵심 문제들

### 문제 1: ESDE_in 편향 (가장 심각)
**증상**: NORMAL처럼 보이는 데이터를 ESDE_in으로 분류
**원인 가설**:
1. 학습 데이터에서 ESDE_in이 180개 파일(~55,800 샘플)로 SGTR(15개, ~4,650) 대비 12배 많음
2. ESDE_in의 초기 패턴이 NORMAL과 유사할 가능성
3. 5에폭 학습이 충분하지 않아 decision boundary가 부정확
4. 학습-테스트 간 반올림 도메인 갭이 clip만으로 완전히 해결되지 않았을 수 있음

**시도할 수 있는 해결책**:
- 100에폭 학습 (현재 진행중) → 더 나은 decision boundary
- GRACE_PERIOD를 5 → 12~15로 확대 (malf_delay 최대 55초 → sec12까지 NORMAL 가능)
- 확정 confidence threshold 도입 (예: 90% 이상일 때만 확정)
- ESDE_in에 대해 더 높은 확정 기준 적용

### 문제 2: 확정 로직 설계
**현재**: 연속 3초 동일 비정상 → 확정 (번복 불가)
**문제점**:
1. GRACE_PERIOD=5이면 sec6~8에서 확정 가능 → malf_delay가 큰 테스트에서 NORMAL 구간에서 오확정
2. 한번 확정되면 번복 불가 → 초기 오판이 치명적
3. confidence 기준이 없음 → 60% 확률로도 확정됨

**개선안**:
- GRACE_PERIOD를 malf_delay 최대값 기반으로 설정 (sec12~15)
- minimum confidence threshold (예: 0.85 이상만 확정)
- 특정 클래스(ESDE_in)는 더 높은 기준 적용
- 확정 후에도 대폭적인 예측 변경이 있으면 재확정 허용?

### 문제 3: 데이터 서브샘플링에 따른 미분 피처 영향
- 원본 학습 데이터: 0.5초 간격 → diff는 0.5초 차이
- 실제 테스트 데이터: 5초 간격 서브샘플링 → 누적 버퍼의 diff는 5초 차이
- **미분 기반 피처의 스케일이 ~10배 차이날 수 있음**
- 이것이 모델 혼란의 원인일 가능성 높음

### 문제 4: 학습-추론 간 불일치
- **학습 시**: 1개 CSV 파일 전체(310행)에 대해 physics feature 계산 → diff가 0.5초 간격
- **추론 시**: 매초 1행씩 누적 → diff가 1초(실시간) 또는 5초(서브샘플링) 간격
- **window_size=3**: 학습 시 연속 3행(1.5초 구간), 추론 시 3초 구간
- 시간 스케일 불일치가 피처 분포를 왜곡할 수 있음

---

## 7. 현재 진행 상황

1. ✅ StandardScaler 폭발 문제 해결 (clip + near-constant 제거)
2. ✅ 학습 데이터 delay 오라벨 해결 (skip_delay_rows=10)
3. ✅ 5에폭 TCN 내부 테스트 99.58%
4. ✅ NORMAL 원본 데이터 정상 분류 확인
5. 🔄 100에폭 TCN 학습 중 (로컬 터미널에서 실행)
6. ❌ 실제 테스트 데이터 ~12% 정확도 (ESDE_in 편향)
7. ❌ 확정 로직 최적화 필요

---

## 8. 핵심 코드 구조

```
NAS/
├── data/
│   ├── data_new/                    # 학습 데이터 (1759 CSV)
│   └── real_test_data/              # 테스트 데이터 (100 시나리오)
│       ├── answers.csv              # 정답
│       └── test{id}_sec{n}.csv      # 각 초별 데이터
├── practice/                        # 핵심 모듈
│   ├── dataloader.py               # 데이터 로딩 + skip_delay_rows
│   ├── feature_method.py           # V2 physics feature (48개)
│   ├── feature_method_v3.py        # V3 phase2 feature (+30개)
│   ├── model.py                    # TCN, CNN, Transformer 등
│   ├── main.py                     # V2 학습 파이프라인
│   └── main_v3.py                  # V3 학습 파이프라인
├── Team N code/
│   ├── models/                     # 학습된 모델 파일
│   ├── data/test{1~100}/           # 테스트 데이터 (대회 폴더 구조)
│   └── py/
│       ├── main.py                 # 대회 제출용 실시간 추론
│       ├── test_local.py           # 단일 테스트 추론
│       └── test_200_real.py        # 100개 배치 테스트
├── useless_features_all.json       # 제거할 113개 피처 목록
├── train_tcn_fix.sh                # 100에폭 학습 스크립트
└── models_9class_v3_fix/           # 학습된 모델 저장
```

---

## 9. 주요 파일 핵심 코드

### 9-1. 실시간 추론 (Team N code/py/main.py) — 확정 로직

```python
class RealtimeInference:
    WINDOW = 3
    CONFIRM_COUNT = 3    # 연속 3초 동일 비정상이면 확정
    GRACE_PERIOD = 5     # 처음 5초는 확정 안 함

    def process_sec(self, x_raw, sec, col_names):
        # 1) 2205 컬럼 → 학습용 188 컬럼 추출
        # 2) 버퍼에 추가
        # 3) Physics V3 피처 변환 (누적 전체)
        # 4) StandardScaler + clip(-10, 10)
        # 5) 마지막 3행 윈도우 → 모델 추론
        # 6) 확정 로직:
        if (not self.confirmed and pred != 0
                and len(self.pred_history) > self.GRACE_PERIOD):
            if len(self.pred_history) >= self.CONFIRM_COUNT:
                recent = self.pred_history[-self.CONFIRM_COUNT:]
                if all(p == pred for p in recent):
                    self.confirmed = True  # 번복 불가!
```

### 9-2. 학습 데이터 로딩 (practice/dataloader.py) — delay 제거

```python
def load_one_csv(file_name, folder_path, include_time=False, skip_delay_rows=0):
    label_id = infer_label_id(file_name)
    df = pd.read_csv(fp)
    # 사고 파일의 앞 N행 제거 (delay 구간 오라벨)
    if skip_delay_rows > 0 and label_id != 0:  # NORMAL이 아닌 경우만
        if len(df) > skip_delay_rows:
            df = df.iloc[skip_delay_rows:].reset_index(drop=True)
    y = np.full((len(df),), label_id, dtype=np.int64)  # 전체 행에 동일 라벨
    return df, y
```

### 9-3. TCN 모델 구조 (practice/model.py)

```python
def build_tcn(window_size, n_features, n_classes, dropout_rate=0.3):
    inputs = Input(shape=(window_size, n_features))  # (3, 266)
    x = tcn_residual_block(inputs, 128, kernel=2, dilation=1)  # 수용영역 3
    x = tcn_residual_block(x, 128, kernel=2, dilation=2)       # 수용영역 7
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, 'relu')(x) → BN → Dropout(0.3)
    outputs = Dense(9, 'softmax')(x)
```

---

## 10. GPT에게 요청하고 싶은 진단/조언

1. **ESDE_in 편향 원인**: 왜 NORMAL 구간 데이터가 ESDE_in으로 분류되는가? 근본 원인은?

2. **학습-추론 시간 스케일 불일치**: 학습은 0.5초 간격, 추론은 실시간 1초(또는 서브샘플링 5초). 미분 피처에 영향? 해결 방법은?

3. **확정 로직 최적화**: GRACE_PERIOD, CONFIRM_COUNT, confidence threshold의 최적값은? 클래스별로 다르게 해야 하나?

4. **데이터 불균형 대응**: NORMAL 4개 vs LOCA 450개 vs SGTR 15개. 현재 augmentation + focal loss + class weight로 대응 중. 더 나은 전략은?

5. **100에폭 학습 후에도 ESDE_in 편향이 지속된다면**: 모델 구조 변경? 피처 추가? 2단계 분류기?

6. **서브샘플링 된 테스트 데이터에 대한 접근**: 원본 310행이 5초 간격으로 60개로 된 건데, 이런 시간 해상도 차이를 어떻게 처리해야 하나?
