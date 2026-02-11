#!/bin/bash
# =======================================
# V3 본학습 스크립트 — subsample=5 + delay_as_normal
# 실제 테스트 97% (5에폭) → 100에폭으로 더 개선 예상
# =======================================
# 사용법: bash train_9class_v3_ss5.sh
# 또는:  nohup bash train_9class_v3_ss5.sh > train_v3_ss5.log 2>&1 &

set -e

PYTHON="/opt/anaconda3/envs/team_6/bin/python3"
DATA="data/data_new"

# 공통 옵션: subsample_stride=5 + delay_as_normal + skip_delay_rows=5
COMMON="--data_folder $DATA --train --use_val --use_class_weight \
  --skip_delay_rows 5 --delay_as_normal --subsample_stride 5 \
  --window_size 3 --stride 1 --seed 0"

echo "============================================"
echo "  V3 본학습 (100에폭, subsample=5, DAN)"
echo "  시작: $(date)"
echo "============================================"

# === TCN (가장 유망) ===
echo ""
echo ">>> TCN 100에폭 시작..."
$PYTHON -m practice.main_v3 \
  --model_folder models_9class_v3_ss5 \
  --train_folder train_results_9class_v3_ss5 \
  --test_folder test_results_9class_v3_ss5 \
  --model_type tcn --epochs 100 \
  $COMMON

echo ""
echo "============================================"
echo "  완료: $(date)"
echo "============================================"
