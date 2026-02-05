#!/bin/bash
# 3초 진단 우승 전략 학습 스크립트

echo "🏆 3초 진단 우승 모델 학습 시작"
echo "=================================="

# Conda 환경 확인
if ! conda env list | grep -q "team_6"; then
    echo "❌ team_6 환경이 없습니다. conda activate team_6 를 실행하세요."
    exit 1
fi

# 1. Physics 피처 + CNN1D (window_size=6)
echo ""
echo "📊 [1/3] Physics + CNN1D (3초 윈도우)"
python -m practice.main \
  --data_folder data/data_new \
  --model_type cnn1d \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  --train

# 2. Diff 피처 + MLP_v2 (빠른 추론)
echo ""
echo "📊 [2/3] Diff + MLP_v2"
python -m practice.main \
  --data_folder data/data_new \
  --model_type mlp_v2 \
  --feature_method diff \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --lr 1e-3 \
  --train

# 3. 앙상블 (최고 성능)
echo ""
echo "📊 [3/3] Ensemble (Physics + window_size=6)"
python -m practice.main \
  --data_folder data/data_new \
  --ensemble \
  --ensemble_method weighted_vote \
  --feature_method physics \
  --window_size 6 \
  --epochs 100 \
  --batch_size 128 \
  --use_val \
  --use_class_weight \
  --train

echo ""
echo "✅ 학습 완료!"
echo ""
echo "📁 저장된 모델:"
ls -lh models/*__model.keras | tail -3
echo ""
echo "📈 결과 확인:"
echo "  - train_results/ (학습 곡선)"
echo "  - test_results/ (혼동 행렬, 클래스별 정확도)"
echo ""
echo "🧪 다음 단계:"
echo "  1. test_results/ 에서 3초 시점 정확도 확인"
echo "  2. 최고 성능 모델로 실시간 테스트"
echo "  3. realtime_inference.py로 3초 진단 테스트"
