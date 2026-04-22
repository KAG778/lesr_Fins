#!/bin/bash
# Monitor 4-stock 2012-2017 pipeline
LOG="exp4.7/logs/run_4stocks_2012_2017_console.log"
PID=$(ps aux | grep "run_4stocks_2012_2017" | grep -v grep | awk '{print $2}')

echo "=========================================="
echo "4-Stock Pipeline Monitor - $(date '+%H:%M:%S')"
echo "=========================================="

if [ -z "$PID" ]; then
    echo "STATUS: Pipeline NOT running"
    if [ -f "exp4.7/result_4.8_ 训练 2012-2014 | 验证 2015-2016 | 测试 2017/test_set_results.pkl" ]; then
        echo "FINAL RESULTS exist!"
    fi
    echo ""
    echo "Last 5 lines:"
    tail -5 "$LOG"
else
    ETIME=$(ps -p "$PID" -o etime= | tr -d ' ')
    echo "STATUS: RUNNING (PID: $PID, elapsed: $ETIME)"

    IT0=$(grep -c "Iteration 0 completed" "$LOG" 2>/dev/null)
    IT1=$(grep -c "Iteration 1 completed" "$LOG" 2>/dev/null)
    IT2=$(grep -c "Iteration 2 completed" "$LOG" 2>/dev/null)
    echo "Iterations: 0=$([ "$IT0" -gt 0 ] 2>/dev/null && echo 'DONE' || echo 'running')  1=$([ "$IT1" -gt 0 ] 2>/dev/null && echo 'DONE' || echo 'pending')  2=$([ "$IT2" -gt 0 ] 2>/dev/null && echo 'DONE' || echo 'pending')"

    CUR_SAMPLE=$(grep "Training sample" "$LOG" | tail -1)
    CUR_EPISODE=$(grep "Episode" "$LOG" | tail -1 | grep -oP 'Episode \d+/\d+' | tail -1)
    echo "Current: $CUR_SAMPLE | $CUR_EPISODE"

    # Show per-ticker latest progress
    echo ""
    echo "Per-ticker latest Episode:"
    for t in TSLA NFLX AMZN MSFT; do
        EP=$(grep "  \[$t\]" "$LOG" | tail -1)
        [ -n "$EP" ] && echo "  $EP"
    done

    # GPU usage
    echo ""
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
        echo "  GPU $line"
    done
fi
echo "=========================================="
