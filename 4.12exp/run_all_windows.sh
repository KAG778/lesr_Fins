#!/bin/bash
#
# 4.12exp: Run all 10 sliding window experiments in parallel
# Layer A+B improvements (random detection + feature diversity)
#
# Usage: bash 4.12exp/run_all_windows.sh
#

set -e

cd /home/wangmeiyi/AuctionNet/lesr

PYTHON=/home/wangmeiyi/miniconda3/bin/python
SCRIPT="4.12exp/run_experiment.py"
LOG_DIR="4.12exp/logs"

mkdir -p "$LOG_DIR"

# Number of parallel jobs (4 GPUs, but each experiment trains sequentially on 1 GPU)
# Run 4 at a time, cycling through GPUs 0-3
MAX_PARALLEL=4
GPU_LIST=(0 1 2 3)

echo "============================================================"
echo "4.12exp: Running 10 sliding window experiments (3-2-1)"
echo "  Layer A: Random reward detection"
echo "  Layer B: Structured feature library for diversity"
echo "  GPUs: ${GPU_LIST[@]}"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Started: $(date)"
echo "============================================================"

run_window() {
    local window=$1
    local gpu=$2
    local config="4.12exp/config_${window}.yaml"
    local log="${LOG_DIR}/run_${window}.log"

    echo "[$(date '+%H:%M:%S')] Starting ${window} on GPU ${gpu}..."

    CUDA_VISIBLE_DEVICES=$gpu $PYTHON "$SCRIPT" --config "$config" > "$log" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ${window} COMPLETED successfully"
    else
        echo "[$(date '+%H:%M:%S')] ${window} FAILED with status ${status}"
    fi
    return $status
}

# Export function for parallel use
export -f run_window
export PYTHON SCRIPT LOG_DIR

# Launch all 10 windows, 4 at a time
active_jobs=()
active_gpus=()

for i in $(seq 1 10); do
    window="W${i}"
    gpu_idx=$(( (i - 1) % ${#GPU_LIST[@]} ))
    gpu=${GPU_LIST[$gpu_idx]}

    run_window "$window" "$gpu" &
    active_jobs+=($!)
    active_gpus+=($gpu)

    echo "  Launched ${window} (PID $!, GPU ${gpu})"

    # If we've hit max parallel, wait for one to finish
    if [ ${#active_jobs[@]} -ge $MAX_PARALLEL ]; then
        # Wait for any one job to finish
        for j in "${!active_jobs[@]}"; do
            if ! kill -0 ${active_jobs[$j]} 2>/dev/null; then
                wait ${active_jobs[$j]}
                unset 'active_jobs[j]'
                unset 'active_gpus[j]'
                # Re-index arrays
                active_jobs=("${active_jobs[@]}")
                active_gpus=("${active_gpus[@]}")
                break
            fi
        done
    fi
done

# Wait for all remaining jobs
echo ""
echo "All 10 experiments launched. Waiting for completion..."
for pid in "${active_jobs[@]}"; do
    wait $pid 2>/dev/null || true
done

echo ""
echo "============================================================"
echo "All experiments completed: $(date)"
echo "============================================================"

# Print summary
echo ""
echo "Results summary:"
for i in $(seq 1 10); do
    window="W${i}"
    log="${LOG_DIR}/run_${window}.log"
    if grep -q "FINAL SUMMARY" "$log" 2>/dev/null; then
        echo "  ${window}: COMPLETED"
    else
        echo "  ${window}: INCOMPLETE or FAILED"
    fi
done
