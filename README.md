# 원전 사고 진단 AI 시스템

원자력발전소 9종 사고를 실시간으로 진단하는 딥러닝 시스템.
CNN+Attention 기반 97.6% 정확도, 0.05ms/sample 추론 속도.

---

## 주요 기능

- **9종 사고 분류**: NORMAL, LOCA(HL/CL/RCP), SGTR(Loop1/2/3), ESDE(in/out)
- **5가지 모델**: MLP, CNN, CNN+Attention, LSTM, Transformer
- **Physics 피처 엔지니어링**: 179 -> 207 피처 (미분, 비대칭, 커플링)
- **XAI 분석**: SHAP 피처 중요도 + Attention 시각화
- **데이터 증강**: Jitter, Scaling, Mixup, 소수 클래스 오버샘플링
- **실시간 추론**: 연속 합의 기반 4-Tier 확정 로직
- **파이프라인 격리**: 모델별 전처리 파일 1:1 매핑 (덮어쓰기 방지)

---

## 프로젝트 구조

```
NAS/
├── practice/                          # 핵심 모듈
│   ├── main.py                        # 학습/평가 파이프라인
│   ├── dataloader.py                  # 데이터 로딩 (런 단위)
│   ├── preprocessing.py               # 전처리 파이프라인 (학습=추론 일치)
│   ├── model.py                       # 5가지 모델 아키텍처
│   ├── feature_method.py              # 피처 엔지니어링 (6종)
│   ├── data_split.py                  # 런 단위 데이터 분할 (누수 방지)
│   ├── augmentation.py                # 시계열 데이터 증강
│   ├── xai.py                         # XAI 분석 (SHAP + Attention)
│   └── utils_plot.py                  # 시각화 유틸
├── competition_interface.py           # 경진대회 실시간 인터페이스
├── realtime_inference.py              # 실시간 추론 (기본)
├── realtime_inference_optimized.py    # 실시간 추론 (최적화)
├── train_3sec_champion.sh             # 학습 스크립트
├── useless_features_all.json          # 불필요 변수 목록 (121개)
├── data/data_new/                     # 센서 데이터 (75개 CSV)
├── models/                            # 학습된 모델 + 전처리 파이프라인
├── train_results/                     # 학습 곡선 (acc, loss)
├── test_results/                      # 혼동 행렬, 분류 리포트
└── xai_results/                       # XAI 시각화 결과
    ├── shap/                          #   SHAP 피처 중요도
    └── attention/                     #   Attention 히트맵
```

---

## 환경 설정

```bash
# conda 가상환경 (Python 3.11)
conda create -n team_6 python=3.11
conda activate team_6

# 패키지 설치
pip install -r requirements.txt
```

---

## 빠른 시작

### 1. 기본 학습

```bash
conda run -n team_6 python -m practice.main \
    --data_folder data/data_new \
    --model_type cnn_attention \
    --feature_method physics \
    --window_size 3 --stride 1 \
    --epochs 100 --batch_size 128 \
    --use_val --use_class_weight \
    --train
```

### 2. 고급 학습 (증강 + 튜닝 전체 적용)

```bash
conda run -n team_6 python -m practice.main \
    --data_folder data/data_new \
    --model_type cnn_attention \
    --feature_method physics \
    --window_size 3 --stride 1 \
    --epochs 100 --batch_size 128 --lr 1e-3 \
    --use_val --use_class_weight \
    --use_augmentation --augment_minority \
    --lr_schedule warmup_cosine \
    --use_adamw --weight_decay 1e-4 \
    --use_focal_loss \
    --early_stopping_patience 25 \
    --train
```

### 3. 5개 모델 일괄 학습

```bash
chmod +x train_3sec_champion.sh
./train_3sec_champion.sh
```

### 4. XAI 분석

```bash
conda run -n team_6 python -m practice.xai \
    --model_dir models \
    --data_folder data/data_new \
    --output_dir xai_results \
    --analysis_type all
```

---

## 사고 유형 (9종)

| ID | 클래스 | 설명 |
|----|--------|------|
| 0 | NORMAL | 정상 운전 |
| 1 | LOCA_HL | 냉각재 상실 - Hot Leg |
| 2 | LOCA_CL | 냉각재 상실 - Cold Leg |
| 3 | LOCA_RCP | 냉각재 상실 - RCP Seal |
| 4 | SGTR_Loop1 | 증기발생기 파열 - Loop 1 |
| 5 | SGTR_Loop2 | 증기발생기 파열 - Loop 2 |
| 6 | SGTR_Loop3 | 증기발생기 파열 - Loop 3 |
| 7 | ESDE_in | 과냉각 사고 - 내부 |
| 8 | ESDE_out | 과냉각 사고 - 외부 |

---

## 모델 옵션

### 아키텍처

| 모델 | 인자 | 특징 |
|------|------|------|
| MLP | `--model_type mlp` | Flat 입력, 빠른 추론 |
| CNN | `--model_type cnn` | 시계열 패턴 추출 |
| CNN+Attention | `--model_type cnn_attention` | 시계열 + 시점 가중치 (추천) |
| LSTM | `--model_type lstm` | 순서 의존성 학습 |
| Transformer | `--model_type transformer` | Self-attention 기반 |

### 피처 엔지니어링

| 방법 | 인자 | 출력 피처 수 | 설명 |
|------|------|-------------|------|
| all | `--feature_method all` | 179 | 원본 (불필요 121개 제거) |
| physics | `--feature_method physics` | 207 | 원본 + 미분/비대칭/커플링 (추천) |
| diff | `--feature_method diff` | 358 | 원본 + 1차 미분 |
| stats | `--feature_method stats` | 537 | 원본 + 이동평균/표준편차 |
| selection | `--feature_method selection` | topk | LGBM 기반 Top-K 선택 |

### 학습 설정

```bash
# 기본 설정
--epochs 100 --batch_size 128 --lr 1e-3 --seed 0

# 검증 + 클래스 가중치
--use_val --use_class_weight

# 데이터 증강
--use_augmentation              # 전체 클래스 (Jitter + Scaling)
--augment_minority              # 소수 클래스 오버샘플링
--minority_classes 8            # 대상 클래스 (기본: ESDE_out)

# 학습률 스케줄링
--lr_schedule warmup_cosine     # Warmup + Cosine Decay
--lr_schedule cosine            # Cosine Decay only

# 옵티마이저
--use_adamw --weight_decay 1e-4 # AdamW (weight decay)

# 손실 함수
--use_focal_loss                # Focal Loss (클래스 불균형)
--label_smoothing 0.1           # Label Smoothing

# 콜백
--early_stopping_patience 25    # EarlyStopping patience
```

---

## 실시간 추론

### 경진대회 인터페이스

```python
from competition_interface import CompetitionSystem

# 시스템 생성 (모델별 파이프라인 자동 로드)
system = CompetitionSystem(
    model_path="models/cnn_attention__feat=physics__....__model.keras",
    window_size=3
)

# 컬럼 매핑 설정
system.set_feature_names(all_feature_names)

# 매 1초마다 호출
for row in sensor_stream:
    result = system.receive_and_predict(row)
    # result = {'results': 'LOCA_HL', 'Diagnostic_time': 5.0, 'Class probabilities': [...]}
```

### 확정 로직 (4-Tier)

사고 진단 확정 시 오탐을 방지하기 위한 연속 합의 + 마진 기반 로직:

| Tier | 조건 | 용도 |
|------|------|------|
| 1 | 확신도 >= 0.90, 연속 2회, 마진 >= 0.30 | 빠른 진단 (명확한 사고) |
| 2 | 확신도 >= 0.70, 연속 3회, 마진 >= 0.15 | 안정적 진단 (일반) |
| 3 | 확신도 >= 0.50, 연속 5회, 평균 >= 0.55 | 보수적 진단 (불확실) |
| 4 | 경과 >= 55초, 확신도 >= 0.40 | 시간 초과 대비 |

---

## XAI (모델 해석성)

### SHAP 분석

```bash
# 전체 분석 (SHAP + Attention)
conda run -n team_6 python -m practice.xai \
    --model_dir models --data_folder data/data_new \
    --output_dir xai_results --analysis_type all
```

출력 파일:
- `xai_results/shap/global_importance_top20.png` - Top 20 피처 중요도
- `xai_results/shap/per_class_importance.png` - 클래스별 피처 중요도
- `xai_results/shap/global_importance_all.csv` - 전체 피처 중요도 순위
- `xai_results/shap/force_plots/` - 개별 예측 설명

### Attention 시각화

- `xai_results/attention/attention_heatmap_all_classes.png` - 클래스별 Attention 히트맵
- `xai_results/attention/per_head_analysis.png` - Head별 시점 집중 패턴
- `xai_results/attention/attention_by_class.png` - 사고 유형별 Attention 비교

---

## 파이프라인 격리 (Goal A)

모델마다 전처리 파이프라인이 1:1로 저장되어 다른 모델 학습 시 덮어쓰기를 방지합니다.

```
models/
├── cnn_attention__feat=physics__...__model.keras
├── cnn_attention__feat=physics__...__scaler.pkl
├── cnn_attention__feat=physics__...__feature_transformer.pkl
├── cnn_attention__feat=physics__...__preprocessing_metadata.json
├── cnn_attention__feat=physics__...__config.json
├── cnn_attention__feat=physics__...__class_mapping.npy
├── cnn__feat=physics__...__model.keras
├── cnn__feat=physics__...__scaler.pkl
└── ...
```

추론 시 모델 경로에서 prefix를 자동 추출하여 올바른 파이프라인을 로드합니다.

---

## 학습 결과

### 생성 파일 구조

```
models/                                          # 모델 + 전처리
├── {run_name}__model.keras
├── {run_name}__scaler.pkl
├── {run_name}__feature_transformer.pkl
├── {run_name}__preprocessing_metadata.json
├── {run_name}__config.json
└── {run_name}__class_mapping.npy

train_results/{run_name}/                        # 학습 곡선
├── acc_vs_epoch.png
└── loss_vs_epoch.png

test_results/{run_name}/                         # 평가 결과
├── test_metrics.txt
├── confusion_matrix.png
└── per_class_accuracy.png

xai_results/                                     # XAI 분석
├── shap/
│   ├── global_importance_top20.png
│   ├── per_class_importance.png
│   ├── global_importance_all.csv
│   └── force_plots/sample_{0..9}.png
└── attention/
    ├── attention_heatmap_all_classes.png
    ├── per_head_analysis.png
    └── attention_by_class.png
```

---

## 데이터

- **위치**: `data/data_new/` (75개 CSV 파일)
- **샘플링**: 1Hz (1초 간격)
- **피처**: 300개 원본 -> 179개 (불필요 121개 자동 제거)
- **사고별 다양한 누출 규모** (leak size) 시뮬레이션 포함

---

## 버전 이력

### v3.0 (경진대회 최적화)
- 모델별 전처리 파이프라인 격리 (덮어쓰기 방지)
- 연속 합의 + 마진 기반 4-Tier 확정 로직 (오탐 최소화)
- 컬럼 mismatch 자동 처리 강화 (순서 보장 + 누락 경고)
- 커스텀 객체(WarmupCosineSchedule) 자동 등록으로 모델 로드 안정화

### v2.0 (XAI + 학습 개선)
- XAI 모듈 (SHAP GradientExplainer + Attention 시각화)
- 데이터 증강 (Jitter, Scaling, Mixup, MinorityOversampling)
- Focal Loss, AdamW, Warmup+Cosine LR 스케줄
- Label Smoothing, EarlyStopping patience 조정
- 97.6% 정확도 달성 (v1 대비 +2.8%p)

### v1.0 (초기)
- 5가지 모델 아키텍처 (MLP/CNN/CNN+Attention/LSTM/Transformer)
- Physics 피처 엔지니어링
- 런 단위 데이터 분할 (누수 방지)
- PreprocessingPipeline (학습=추론 일치)
- 불필요 변수 121개 자동 제거
