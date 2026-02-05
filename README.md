# NAS (Nuclear Accident Scenario) Classification

원자력 발전소 사고 시나리오 분류를 위한 딥러닝 모델 프로젝트입니다.

**🏆 목표: 3초 안에 사고 유형 정확하게 진단**

## 📚 문서 가이드
- **빠른 시작**: 이 문서 (README.md)
- **3초 전략 요약**: [3SEC_SUMMARY.md](3SEC_SUMMARY.md) ⭐
- **상세 우승 전략**: [WINNING_STRATEGY.md](WINNING_STRATEGY.md)
- **프로젝트 구조**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **변경 이력**: [CHANGELOG.md](CHANGELOG.md)

## 환경 설정

이 프로젝트는 `team_6` conda 환경에서 실행됩니다.

```bash
conda activate team_6
```

## 데이터 구조

### 9개 클래스 체계

이 프로젝트는 9개 클래스를 지원합니다:
- `NORMAL`: 정상 상태
- `LOCA_HL`: Hot Leg LOCA (냉각재 손실 사고 - Hot Leg)
- `LOCA_CL`: Cold Leg LOCA (냉각재 손실 사고 - Cold Leg)
- `LOCA_RCPCSEAL`: RCP Seal LOCA (냉각재 손실 사고 - RCP Seal)
- `SGTR_SG1`: 증기 발생기 1번 세관 파열
- `SGTR_SG2`: 증기 발생기 2번 세관 파열
- `SGTR_SG3`: 증기 발생기 3번 세관 파열
- `MSLB_in`: 주증기관 파열 (내부)
- `MSLB_out`: 주증기관 파열 (외부)

### 파일명 규칙

각 CSV 파일의 prefix로 라벨을 구분합니다:
- `NORMAL_*.csv`: 정상 상태
- `LOCA_HL_*.csv`: Hot Leg LOCA
- `LOCA_CL_*.csv`: Cold Leg LOCA
- `LOCA_RCPCSEAL_*.csv`: RCP Seal LOCA
- `SGTR_SG1_*.csv`: SG1 파열
- `SGTR_SG2_*.csv`: SG2 파열
- `SGTR_SG3_*.csv`: SG3 파열
- `MSLBIN_*.csv` 또는 `MSLB_in_*.csv`: MSLB 내부
- `MSLBOUT_*.csv` 또는 `MSLB_out_*.csv`: MSLB 외부

### 현재 데이터

현재 데이터 위치: `data/data_new/`

**현재 가용한 클래스 (5개):**
- NORMAL: 3개 파일
- LOCA_CL: 9개 파일 (세부 분류되지 않은 LOCA_*.csv 파일 포함)
- SGTR_SG1: 3개 파일 (세부 분류되지 않은 SGTR_*.csv 파일 포함)
- MSLB_in: 27개 파일
- MSLB_out: 27개 파일

**누락된 클래스 (4개):**
- LOCA_HL (0개)
- LOCA_RCPCSEAL (0개)
- SGTR_SG2 (0개)
- SGTR_SG3 (0개)

데이터 로드 시 자동으로 클래스 분포를 확인하고 누락된 클래스를 경고합니다.

## 사용법

### 기본 학습

```bash
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp \
  --epochs 100 \
  --batch_size 128 \
  --train
```

### 모델 타입

- `mlp`: Multi-Layer Perceptron (기본값)
- `mlp_v2`: 개선된 MLP
- `cnn`: 2D CNN
- `cnn1d`: 1D CNN (시계열 데이터용)
- `lstm`: LSTM
- `hybrid`: CNN1D + LSTM

### 시계열 모델 학습 (윈도우 사용)

```bash
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn1d \
  --window_size 10 \
  --epochs 100 \
  --batch_size 128 \
  --train
```

### 앙상블 모드

```bash
python -m practice.main \
  --data_folder data/data_new \
  --ensemble \
  --ensemble_method soft_vote \
  --use_val \
  --window_size 10 \
  --epochs 100 \
  --batch_size 128 \
  --train
```

### 주요 옵션

- `--data_folder`: 데이터 폴더 경로 (필수)
- `--model_type`: 모델 종류 선택
- `--feature_method`: 피처 추출 방법 (all, change, selection, diff, stats, physics)
- `--group_size`: 그룹 크기 (기본값: 10)
- `--use_val`: 검증 셋 사용
- `--epochs`: 학습 에포크 수 (기본값: 100)
- `--batch_size`: 배치 크기 (기본값: 128)
- `--lr`: 학습률 (기본값: 1e-3)
- `--use_class_weight`: 클래스 불균형 처리
- `--window_size`: 시계열 모델용 윈도우 크기 (기본값: 10)
- `--ensemble`: 앙상블 모드 활성화
- `--ensemble_method`: 앙상블 방법 (soft_vote, weighted_vote, stacking)
- `--seed`: 랜덤 시드 (기본값: 0)

### 학습된 모델 테스트

```bash
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp \
  --epochs 100 \
  --batch_size 128
```
(--train 플래그를 제거하면 저장된 모델을 로드하여 테스트합니다)

## 출력 결과

### 저장 위치
- 모델: `models/`
- 학습 결과: `train_results/`
- 테스트 결과: `test_results/`

### 생성되는 파일
- `*_model.keras`: 저장된 모델
- `acc_vs_epoch.png`: 학습 정확도 그래프
- `loss_vs_epoch.png`: 학습 손실 그래프
- `confusion_matrix.png`: 혼동 행렬
- `per_class_accuracy.png`: 클래스별 정확도
- `test_metrics.txt`: 테스트 메트릭 (정확도, 손실, 추론 시간 등)

## 프로젝트 구조

```
NAS/
├── practice/
│   ├── main.py              # 메인 실행 스크립트
│   ├── dataloader.py        # 데이터 로드 및 전처리
│   ├── model.py             # 모델 정의
│   ├── feature_method.py    # 피처 엔지니어링
│   ├── data_split.py        # 데이터 분할
│   └── utils_plot.py        # 시각화 유틸리티
├── data/
│   └── data_new/            # 데이터 폴더 (9개 클래스 중 5개 가용)
├── models/                  # 저장된 모델
├── train_results/           # 학습 결과
├── test_results/            # 테스트 결과
├── quick_test.sh            # 빠른 테스트 스크립트
├── train_best_models.sh     # 최적 모델 학습 스크립트
└── realtime_inference.py    # 실시간 진단 시스템
```

## 예제

### 1. MLP 모델 빠른 테스트 (2 에포크)
```bash
conda activate team_6
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp \
  --epochs 2 \
  --batch_size 128 \
  --train
```

### 2. CNN1D 모델 전체 학습
```bash
conda activate team_6
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn1d \
  --window_size 10 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --train
```

### 3. 앙상블 학습
```bash
conda activate team_6
python -m practice.main \
  --data_folder data/data_new \
  --ensemble \
  --ensemble_method weighted_vote \
  --use_val \
  --window_size 10 \
  --epochs 100 \
  --batch_size 128 \
  --train
```

## 🏆 3초 진단 우승 전략

### 목표: 3초 안에 사고 유형 정확하게 진단

**핵심 전략**:
1. **변화율 피처 사용** (diff, physics) - 급격한 변화 포착
2. **작은 윈도우** (window_size=6) - 3초 데이터만 사용
3. **공격적 threshold** (65%) - 빠른 진단
4. **앙상블 모델** - 높은 정확도

### 빠른 시작
```bash
# 3초 진단 최적 모델 학습
chmod +x train_3sec_champion.sh
./train_3sec_champion.sh
```

### 수동 실행
```bash
# Physics 피처 + CNN1D (추천)
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn1d \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --use_val \
  --use_class_weight \
  --train

# 앙상블 (최고 성능)
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

자세한 내용은 [WINNING_STRATEGY.md](WINNING_STRATEGY.md) 참고

---

## 새로운 기능 (9개 클래스 확장 지원)

### 클래스 매핑 자동 저장
- 학습 시 사용된 클래스 정보가 자동으로 저장됩니다
- 파일명: `{model_name}__class_mapping.npy`
- 누락된 클래스가 있어도 9개 클래스 구조 유지

### 누락 클래스 처리
실시간 추론 시 학습되지 않은 클래스 예측 시 자동 재매핑:
- LOCA_HL(1) → LOCA_CL(2)
- LOCA_RCPCSEAL(3) → LOCA_CL(2)
- SGTR_SG2(5) → SGTR_SG1(4)
- SGTR_SG3(6) → SGTR_SG1(4)

### 개선된 시각화
- Confusion Matrix: 9개 클래스 전체 표시, 누락 클래스는 회색
- Per-class Accuracy: 누락 클래스 구분 표시

### 경진대회 인터페이스
```bash
python competition_interface.py \
  --model models/your_model.keras \
  --class_mapping models/your_model__class_mapping.npy \
  --timeout 60 \
  --interval 5
```

## 참고 사항

- 학습 시 반드시 `team_6` conda 환경을 활성화해야 합니다.
- 시계열 모델 (cnn1d, lstm, hybrid)은 `--window_size` 옵션이 필요합니다.
- 앙상블 모드는 `--use_val` 옵션이 필수입니다.
- 데이터 폴더 경로는 상대 경로 또는 절대 경로를 사용할 수 있습니다.
- 현재 5개 클래스만 학습 가능하나, 9개 클래스 구조로 설계되어 데이터 추가 시 즉시 확장 가능합니다.
