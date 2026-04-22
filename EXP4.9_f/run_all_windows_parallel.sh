#!/bin/bash
cd /home/wangmeiyi/AuctionNet/lesr/EXP4.9_f
PYTHON=/home/wangmeiyi/miniconda3/bin/python3
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== EXP4.9_f: Starting ALL 10 windows in parallel ==="
echo "$(date)"

PIDS=""
for w in 1 2 3 4 5 6 7 8 9 10; do
    config="config_W${w}.yaml"
    log="$LOG_DIR/W${w}.log"
    echo "[W${w}] Starting... $(date)"
    $PYTHON run_window.py --config "$config" > "$log" 2>&1 &
    PIDS="$PIDS $!"
done

echo "[ALL] All 10 launched, PIDs:$PIDS"
echo "Waiting for all to finish..."

for pid in $PIDS; do
    wait $pid || true
done

echo "=== ALL DONE ==="
echo "$(date)"

# Summary
echo ""
echo "=== Results ==="
for w in 1 2 3 4 5 6 7 8 9 10; do
    log="$LOG_DIR/W${w}.log"
    if grep -q "Done" "$log" 2>/dev/null; then
        echo "[W${w}] SUCCESS"
    else
        status=$(tail -1 "$log" 2>/dev/null)
        echo "[W${w}] STATUS: $status"
    fi
done
