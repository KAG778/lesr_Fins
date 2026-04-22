#!/bin/bash
cd /home/wangmeiyi/AuctionNet/lesr/EXP4.9_f
PYTHON=/home/wangmeiyi/miniconda3/bin/python3
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== EXP4.9_f: Starting 10 windows ==="
echo "$(date)"

run_window() {
    local wid=$1
    local config="config_W${wid}.yaml"
    local log="$LOG_DIR/W${wid}.log"
    echo "[W${wid}] Starting... $(date)"
    $PYTHON run_window.py --config "$config" > "$log" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[W${wid}] DONE $(date)"
    else
        echo "[W${wid}] FAILED (exit $status) $(date)"
    fi
    return $status
}

# Batch 1: W1-W4
echo "--- Batch 1: W1-W4 ---"
run_window 1 &
run_window 2 &
run_window 3 &
run_window 4 &
wait || true

# Batch 2: W5-W8
echo "--- Batch 2: W5-W8 ---"
run_window 5 &
run_window 6 &
run_window 7 &
run_window 8 &
wait || true

# Batch 3: W9-W10
echo "--- Batch 3: W9-W10 ---"
run_window 9 &
run_window 10 &
wait || true

echo "=== ALL DONE ==="
echo "$(date)"
