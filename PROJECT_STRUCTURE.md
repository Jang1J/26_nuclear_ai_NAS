# NAS 프로젝트 구조

## 📁 디렉토리 구조

```
NAS/
├── 📄 문서
│   ├── README.md                    # 프로젝트 개요 및 사용법
│   ├── 3SEC_SUMMARY.md             # 3초 진단 전략 요약
│   ├── WINNING_STRATEGY.md         # 상세 우승 전략
│   ├── CHANGELOG.md                # 변경 이력
│   └── PROJECT_STRUCTURE.md        # 이 문서
│
├── 🐍 핵심 코드
│   ├── practice/                   # 메인 패키지
│   │   ├── __init__.py
│   │   ├── main.py                 # 학습/테스트 메인 스크립트
│   │   ├── model.py                # 모델 정의 (MLP, CNN, LSTM, Ensemble)
│   │   ├── dataloader.py           # 데이터 로드 및 전처리
│   │   ├── feature_method.py       # 피처 엔지니어링 (physics 포함)
│   │   ├── data_split.py           # 데이터 분할 로직
│   │   └── utils_plot.py           # 시각화 (Confusion Matrix 등)
│   │
│   ├── realtime_inference.py       # 실시간 진단 시스템 (3초 최적화)
│   └── competition_interface.py    # 경진대회 평가 인터페이스
│
├── 🚀 실행 스크립트
│   └── train_3sec_champion.sh      # 3초 우승 모델 학습 스크립트
│
├── 📊 데이터 & 결과 (gitignore)
│   ├── data/                       # 학습 데이터
│   ├── models/                     # 저장된 모델
│   ├── train_results/              # 학습 곡선
│   └── test_results/               # 테스트 결과
│
└── ⚙️ 설정
    ├── .gitignore
    └── .claude/settings.local.json
```

---

## 📝 주요 파일 설명

### 문서

| 파일 | 용도 |
|------|------|
| `README.md` | 전체 프로젝트 개요, 빠른 시작 가이드 |
| `3SEC_SUMMARY.md` | 3초 진단 핵심 전략 요약 (빠른 참고용) |
| `WINNING_STRATEGY.md` | 상세 우승 전략, 하이퍼파라미터, 테스트 계획 |
| `CHANGELOG.md` | 주요 변경사항 및 버전 이력 |

### 핵심 코드

#### practice/ 패키지

| 파일 | 역할 | 주요 기능 |
|------|------|---------|
| `main.py` | 메인 실행 | 학습/테스트, 단일 모델 & 앙상블 |
| `model.py` | 모델 정의 | MLP, CNN1D, LSTM, Hybrid 구현 |
| `dataloader.py` | 데이터 로더 | 9개 클래스 자동 매핑, 슬라이딩 윈도우 |
| `feature_method.py` | 피처 엔지니어링 | physics (28개), diff, stats 등 |
| `data_split.py` | 데이터 분할 | train/val/test, 클래스 가중치 |
| `utils_plot.py` | 시각화 | Confusion Matrix, Per-class Accuracy |

#### 실시간 시스템

| 파일 | 역할 | 특징 |
|------|------|------|
| `realtime_inference.py` | 실시간 진단 | 3초 공격적 threshold, 누락 클래스 처리 |
| `competition_interface.py` | 경진대회 연동 | 데이터 송수신 템플릿 |

### 실행 스크립트

| 파일 | 용도 |
|------|------|
| `train_3sec_champion.sh` | 3초 최적화 모델 3종 자동 학습 |

---

## 🎯 주요 기능

### 1. 9개 클래스 지원
- NORMAL, LOCA (3종), SGTR (3종), MSLB (2종)
- 자동 클래스 매핑 저장/로드
- 누락 클래스 자동 재매핑

### 2. 다양한 모델
- MLP, MLP_v2
- CNN 2D/1D
- LSTM
- CNN1D + LSTM Hybrid
- Ensemble (3모델 조합)

### 3. Physics 피처 (28개)
- 1차/2차 미분 (16개)
- 루프 비대칭 (7개)
- 물리 관계 (5개)

### 4. 3초 진단 최적화
- 공격적 threshold (65%)
- 초고확신 즉시 확정 (85%)
- 최소 샘플 수 제한

---

## 🚀 사용법

### 빠른 시작

```bash
# 환경 활성화
conda activate team_6

# 3초 우승 모델 학습
./train_3sec_champion.sh
```

### 수동 실행

```bash
# Physics 피처 + CNN1D
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn1d \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --use_val \
  --use_class_weight \
  --train

# 앙상블
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

### 실시간 테스트

```bash
python realtime_inference.py \
  --model models/cnn1d__feat=physics__model.keras \
  --data data/data_new \
  --feature physics \
  --window 6 \
  --class_mapping models/cnn1d__feat=physics__class_mapping.npy
```

---

## 📊 결과 파일

### models/
- `{model_name}__model.keras`: 저장된 모델
- `{model_name}__class_mapping.npy`: 학습 클래스 정보

### train_results/{run_name}/
- `acc_vs_epoch.png`: 학습 정확도 곡선
- `loss_vs_epoch.png`: 학습 손실 곡선

### test_results/{run_name}/
- `confusion_matrix.png`: 혼동 행렬 (9x9)
- `per_class_accuracy.png`: 클래스별 정확도
- `test_metrics.txt`: 정확도, 손실, 추론 시간 등

---

## 🔧 설정

### .gitignore
- 데이터 폴더 제외
- 모델 파일 제외 (용량 큼)
- 결과 폴더 제외
- 캐시 파일 제외

### Conda 환경
- 환경명: `team_6`
- 주요 패키지: tensorflow, numpy, pandas, scikit-learn, lightgbm

---

## 📈 현재 상태

### 데이터
- ✅ 5개 클래스 가용 (NORMAL, LOCA_CL, SGTR_SG1, MSLB_in, MSLB_out)
- ❌ 4개 클래스 누락 (LOCA_HL, LOCA_RCPCSEAL, SGTR_SG2, SGTR_SG3)

### 기능
- ✅ 9개 클래스 구조 완성
- ✅ 3초 진단 최적화 완료
- ✅ Physics 피처 구현 완료
- ⏳ 경진대회 인터페이스 (프로토콜 대기)

---

## 🎯 다음 단계

1. **즉시**: `train_3sec_champion.sh` 실행
2. **테스트**: test_results에서 3초 정확도 확인
3. **데이터**: 누락된 4개 클래스 추가
4. **최종**: 9개 클래스로 재학습 및 경진대회 제출

---

## 📞 참고

- 자세한 사용법: `README.md`
- 3초 전략: `3SEC_SUMMARY.md`
- 상세 전략: `WINNING_STRATEGY.md`
- 변경 이력: `CHANGELOG.md`
