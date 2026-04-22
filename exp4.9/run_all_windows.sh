#!/bin/bash
# Exp4.9: Run all 10 windows in parallel (one per GPU if available)
# Usage: bash exp4.9/run_all_windows.sh
# Or sequential: bash exp4.9/run_all_windows.sh --sequential

cd /home/wangmeiyi/AuctionNet/lesr

MODE="${1:-parallel}"

echo "============================================"
echo "Exp4.9: Regime-Conditioned LESR - 10 Windows"
echo "Mode: $MODE"
echo "Start: $(date)"
echo "============================================"

if [ "$MODE" == "--sequential" ]; then
    # Sequential execution (safer, easier to debug)
    for i in $(seq 1 10); do
        echo ""
        echo "========== Window $i =========="
        python exp4.9/run_window.py --config exp4.9/config_W${i}.yaml 2>&1 | tee -a exp4.9/logs/W${i}.log
    done
else
    # Parallel execution (faster, one window per background process)
    for i in $(seq 1 10); do
        echo "Starting Window $i..."
        nohup python exp4.9/run_window.py --config exp4.9/config_W${i}.yaml > exp4.9/logs/W${i}.log 2>&1 &
    done
    echo ""
    echo "All 10 windows launched. Monitor with:"
    echo "  tail -f exp4.9/logs/W*.log"
    echo ""
    echo "Check status:"
    echo "  bash exp4.9/monitor_all.sh"
fi

echo "End: $(date)"
