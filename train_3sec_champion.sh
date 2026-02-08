#!/bin/bash
# 3초 진단 모델 학습 스크립트 (v3)
# window_size=3 (3초 @1초 샘플링), stride=1 (1초 간격)
# v3: 데이터 증강, Focal Loss, AdamW, Warmup+Cosine LR 추가
#
# 사용법:
#   chmod +x train_3sec_champion.sh
#   ./train_3sec_champion.sh
#
# 환경: conda activate team_6

CONDA_ENV="team_6"
DATA="data/data_new"

# 공통 v3 옵션 (증강 + 고급 튜닝)
V3_OPTS="--use_augmentation --augment_minority \
  --lr_schedule warmup_cosine \
  --use_adamw --weight_decay 1e-4 \
  --use_focal_loss \
  --early_stopping_patience 25"

echo "========================================"
echo "  3초 진단 모델 학습 (v3)"
echo "  - 모델: MLP, CNN(Multi-Scale), CNN+Att, TCN, Transformer"
echo "  - 증강: Jitter + Scaling + MinorityOversampling"
echo "  - 옵티마이저: AdamW (weight_decay=1e-4)"
echo "  - 손실: Focal Loss (gamma=2.0)"
echo "  - LR: Warmup + Cosine Decay"
echo "========================================"

# 1. MLP (빠른 추론)
echo ""
echo "[1/5] MLP"
conda run -n $CONDA_ENV python -m practice.main \
  --data_folder $DATA \
  --model_type mlp \
  \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  $V3_OPTS \
  --train

# 2. CNN (3초 윈도우)
echo ""
echo "[2/5] CNN (3초 윈도우)"
conda run -n $CONDA_ENV python -m practice.main \
  --data_folder $DATA \
  --model_type cnn \
  \
  --window_size 3 --stride 1 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  $V3_OPTS \
  --train

# 3. CNN + Attention (추천 모델)
echo ""
echo "[3/5] CNN + Attention (추천)"
conda run -n $CONDA_ENV python -m practice.main \
  --data_folder $DATA \
  --model_type cnn_attention \
  \
  --window_size 3 --stride 1 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  $V3_OPTS \
  --train

# 4. TCN (LSTM 대체)
echo ""
echo "[4/5] TCN"
conda run -n $CONDA_ENV python -m practice.main \
  --data_folder $DATA \
  --model_type tcn \
  \
  --window_size 3 --stride 1 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  $V3_OPTS \
  --train

# 5. Transformer
echo ""
echo "[5/5] Transformer"
conda run -n $CONDA_ENV python -m practice.main \
  --data_folder $DATA \
  --model_type transformer \
  \
  --window_size 3 --stride 1 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  $V3_OPTS \
  --train

echo ""
echo "========================================"
echo "  학습 완료!"
echo "========================================"
echo ""
echo "저장된 모델:"
ls -lh models/*__model.keras 2>/dev/null | tail -5
echo ""
echo "결과 확인:"
echo "  - train_results/ (학습 곡선)"
echo "  - test_results/  (혼동 행렬, 분류 리포트)"
echo ""
echo "XAI 분석 실행:"
echo "  conda run -n $CONDA_ENV python -m practice.xai \\"
echo "    --model_dir models --data_folder $DATA \\"
echo "    --output_dir xai_results --analysis_type all"
