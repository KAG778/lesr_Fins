#!/bin/bash
# Parallel run: 5 windows x 2 versions (V1, V2) = 10 experiments
# 4 GPUs available: run 4 jobs at a time
# V1 = code transfer + retrain on test first 50%, eval last 50%
# V2 = fine-tune on test first 20%, eval last 80%

set -e

BASE_DIR="/home/wangmeiyi/AuctionNet/lesr"
V1_DIR="${BASE_DIR}/组合优化_ppo_策略迁移_v1"
V2_DIR="${BASE_DIR}/组合优化_ppo_策略迁移_v2"

WINDOWS=(W1 W2 W3 W4 W5)

# GPU assignment: round-robin across 4 GPUs
run_job() {
    local version=$1
    local window=$2
    local gpu=$3
    local dir="${BASE_DIR}/组合优化_ppo_策略迁移_${version}"

    echo "[START] V${version} ${window} on GPU ${gpu}"
    cd "${dir}"
    CUDA_VISIBLE_DEVICES=${gpu} nohup /home/wangmeiyi/miniconda3/bin/python main.py \
        --config "configs/config_${window}.yaml" \
        --experiment_name "window_${window}" \
        > "results/${window}.log" 2>&1
    echo "[DONE] V${version} ${window} on GPU ${gpu}"
}

# Run all V1 windows (4 parallel, then 1)
echo "=========================================="
echo "Starting V1 experiments (5 windows)"
echo "=========================================="

gpu=0
for w in "${WINDOWS[@]:0:4}"; do
    run_job "v1" "$w" "$gpu" &
    gpu=$((gpu + 1))
done
wait

run_job "v1" "W5" "0" &
wait

echo "=========================================="
echo "V1 complete. Starting V2 experiments"
echo "=========================================="

# Run all V2 windows (4 parallel, then 1)
gpu=0
for w in "${WINDOWS[@]:0:4}"; do
    run_job "v2" "$w" "$gpu" &
    gpu=$((gpu + 1))
done
wait

run_job "v2" "W5" "0" &
wait

echo "=========================================="
echo "All 10 experiments complete!"
echo "=========================================="
echo "Results:"
for ver in v1 v2; do
    for w in "${WINDOWS[@]}"; do
        dir="${BASE_DIR}/组合优化_ppo_策略迁移_${ver}/results/window_${w}"
        if [ -f "${dir}/final_comparison.json" ]; then
            echo "  V${ver} ${w}: DONE"
        else
            echo "  V${ver} ${w}: INCOMPLETE"
        fi
    done
done
