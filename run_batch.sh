#!/bin/bash
# Batch experiment runner
# Usage: bash run_batch.sh <gpu_id> <version> <window>

export CUDA_VISIBLE_DEVICES=$1
VERSION=$2
WINDOW=$3

if [ "$VERSION" = "v1" ]; then
  DIR="/home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_策略迁移_v1"
  cd "$DIR"
  mkdir -p results/v1
  /home/wangmeiyi/miniconda3/bin/python -u -c "
import sys, os
sys.path.insert(0, os.getcwd())
from core.lesr_controller import LESRController
import yaml
with open('configs/config_${WINDOW}.yaml') as f:
    config = yaml.safe_load(f)
ctrl = LESRController(config, experiment_dir='results/v1/${WINDOW}')
ctrl.run()
print('\n=== V1 ${WINDOW} DONE ===')
" 2>&1 | tee results/v1/${WINDOW}.log

elif [ "$VERSION" = "v2" ]; then
  DIR="/home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_策略迁移_v2"
  cd "$DIR"
  mkdir -p results/v2
  /home/wangmeiyi/miniconda3/bin/python -u -c "
import sys, os
sys.path.insert(0, os.getcwd())
from core.lesr_controller import LESRController
import yaml
with open('configs/config_${WINDOW}.yaml') as f:
    config = yaml.safe_load(f)
ctrl = LESRController(config, experiment_dir='results/v2/${WINDOW}')
ctrl.run()
print('\n=== V2 ${WINDOW} DONE ===')
" 2>&1 | tee results/v2/${WINDOW}.log

elif [ "$VERSION" = "bl_v1a" ]; then
  DIR="/home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_熊市维护"
  cd "$DIR"
  /home/wangmeiyi/miniconda3/bin/python -u scripts/run_baseline_v1.py \
    --config "configs/config_${WINDOW}.yaml" --name "baseline_v1_${WINDOW}"

elif [ "$VERSION" = "bl_v1b" ]; then
  DIR="/home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_熊市维护"
  cd "$DIR"
  /home/wangmeiyi/miniconda3/bin/python -u scripts/run_baseline_v1b.py \
    --config "configs/config_${WINDOW}.yaml" --name "baseline_v1b_${WINDOW}"

elif [ "$VERSION" = "bl_v2" ]; then
  DIR="/home/wangmeiyi/AuctionNet/lesr/组合优化_ppo_熊市维护"
  cd "$DIR"
  /home/wangmeiyi/miniconda3/bin/python -u scripts/run_baseline_v2.py \
    --config "configs/config_${WINDOW}.yaml" --name "baseline_v2_${WINDOW}"
fi
