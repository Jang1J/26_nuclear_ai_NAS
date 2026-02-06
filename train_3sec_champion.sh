#!/bin/bash
# 3초 진단 학습 스크립트 (1초 샘플링 데이터 기준)
# window_size=3 (3초 @1초 샘플링), stride=1 (1초 간격)

echo "3초 진단 모델 학습 시작"
echo "=================================="

# 1. MLP (빠른 추론)
echo ""
echo "[1/5] MLP"
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp \
  --feature_method physics \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  --train

# 2. CNN (3초 윈도우)
echo ""
echo "[2/5] CNN (3초 윈도우)"
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn \
  --feature_method physics \
  --window_size 3 \
  --stride 1 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  --train

# 3. CNN + Attention
echo ""
echo "[3/5] CNN + Attention"
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn_attention \
  --feature_method physics \
  --window_size 3 \
  --stride 1 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  --train

# 4. LSTM
echo ""
echo "[4/5] LSTM"
python -m practice.main \
  --data_folder data/data_new \
  --model_type lstm \
  --feature_method physics \
  --window_size 3 \
  --stride 1 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  --train

# 5. Transformer
echo ""
echo "[5/5] Transformer"
python -m practice.main \
  --data_folder data/data_new \
  --model_type transformer \
  --feature_method physics \
  --window_size 3 \
  --stride 1 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  --train

echo ""
echo "학습 완료!"
echo ""
echo "저장된 모델:"
ls -lh models/*__model.keras | tail -5
echo ""
echo "결과 확인:"
echo "  - train_results/ (학습 곡선)"
echo "  - test_results/ (혼동 행렬, 클래스별 정확도)"
