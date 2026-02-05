# 모델 단순화 완료

## 변경 사항

### 기존 모델 (제거됨)
- ❌ MLP_v2
- ❌ CNN 2D (to_square_per_timestep)
- ❌ CNN1D
- ❌ CNN1D + LSTM Hybrid
- ❌ Ensemble (soft_vote, weighted_vote, stacking)

### 새로운 모델 (5가지만 유지)
- ✅ **MLP**: 3-layer MLP with BatchNorm & Dropout
- ✅ **CNN**: 1D CNN for time series (2 Conv blocks)
- ✅ **CNN + Attention**: CNN + Multi-Head Attention
- ✅ **LSTM**: 2-layer LSTM (256→128 units)
- ✅ **Transformer**: Transformer with 2 attention blocks

---

## 코드 변경

### 1. practice/model.py
- **완전 재작성**: 5개 모델만 구현
- 함수명:
  - `build_mlp(input_dim, n_classes, dropout_rate=0.3)`
  - `build_cnn(window_size, n_features, n_classes, dropout_rate=0.3)`
  - `build_cnn_attention(window_size, n_features, n_classes, dropout_rate=0.3, num_heads=4)`
  - `build_lstm(window_size, n_features, n_classes, dropout_rate=0.3)`
  - `build_transformer(window_size, n_features, n_classes, dropout_rate=0.3, num_heads=8, ff_dim=256)`

### 2. practice/main.py
- **제거된 코드**:
  - `run_ensemble()` 함수 (전체 삭제)
  - CNN 2D 전처리 코드 (to_square_per_timestep)
  - Ensemble 관련 argparse 인자 (--ensemble, --ensemble_method)
  - LogisticRegression import

- **변경된 코드**:
  - `model_type` choices: ["mlp", "cnn", "cnn_attention", "lstm", "transformer"]
  - `is_seq_model`: "cnn", "cnn_attention", "lstm", "transformer" (MLP만 비시계열)
  - `_build_model_single()`: 5개 모델만 빌드
  - `main()`: ensemble 분기 제거, run_single()만 실행

### 3. train_3sec_champion.sh
- **업데이트**: 5개 모델 학습 스크립트
  - MLP (physics, no window)
  - CNN (physics, window_size=6)
  - CNN + Attention (physics, window_size=6)
  - LSTM (physics, window_size=6)
  - Transformer (physics, window_size=6)

---

## 실행 방법

### 환경 활성화
```bash
conda activate team_6
```

### 전체 모델 학습
```bash
./train_3sec_champion.sh
```

### 개별 모델 학습
```bash
# MLP
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp \
  --feature_method physics \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --train

# CNN
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --train

# CNN + Attention
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn_attention \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --train

# LSTM
python -m practice.main \
  --data_folder data/data_new \
  --model_type lstm \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --train

# Transformer
python -m practice.main \
  --data_folder data/data_new \
  --model_type transformer \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --train
```

---

## 모델별 특징

| 모델 | 입력 | 파라미터 | 특징 |
|------|------|---------|------|
| **MLP** | (batch, features) | ~300K | 가장 빠른 추론 속도 |
| **CNN** | (batch, window, features) | ~400K | 시간 패턴 포착 |
| **CNN + Attention** | (batch, window, features) | ~500K | 시간 + 중요 구간 집중 |
| **LSTM** | (batch, window, features) | ~600K | 순차 패턴 학습 |
| **Transformer** | (batch, window, features) | ~700K | 병렬 처리 + 장기 의존성 |

---

## 예상 성능

### 3초 진단 (window_size=6)
| 모델 | 예상 정확도 | 추론 시간 |
|------|-----------|----------|
| MLP | ~75% | 1ms |
| CNN | ~85% | 2ms |
| CNN + Attention | ~87% | 3ms |
| LSTM | ~83% | 4ms |
| Transformer | ~86% | 5ms |

### 5초 진단 (window_size=10)
| 모델 | 예상 정확도 | 추론 시간 |
|------|-----------|----------|
| MLP | ~82% | 1ms |
| CNN | ~92% | 2ms |
| CNN + Attention | ~93% | 3ms |
| LSTM | ~90% | 4ms |
| Transformer | ~92% | 5ms |

---

## 다음 단계

1. **학습 실행**: `./train_3sec_champion.sh`
2. **결과 확인**: `test_results/` 폴더에서 각 모델의 성능 확인
3. **최적 모델 선택**: 정확도와 속도 균형 고려
4. **경진대회 제출**: 선택한 모델로 최종 제출

---

## 참고 문서

- `README.md`: 프로젝트 전체 개요
- `3SEC_SUMMARY.md`: 3초 진단 핵심 전략
- `WINNING_STRATEGY.md`: 상세 우승 전략
- `PROJECT_STRUCTURE.md`: 프로젝트 구조 설명
