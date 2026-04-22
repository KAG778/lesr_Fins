#!/bin/bash
# Monitor NFLX pipeline progress
LOG="exp4.7/logs/run_nflx_console.log"
PID=$(ps aux | grep "run_nflx" | grep -v grep | awk '{print $2}')

echo "=========================================="
echo "NFLX Pipeline Monitor - $(date '+%H:%M:%S')"
echo "=========================================="

if [ -z "$PID" ]; then
    echo "STATUS: Pipeline NOT running (PID not found)"
    echo ""
    echo "Last 5 lines:"
    tail -5 "$LOG"
    echo ""
    # Check if final results exist
    if [ -f "exp4.7/results_nflx_only_new/test_set_results.pkl" ]; then
        echo "FINAL RESULTS exist!"
    elif [ -f "exp4.7/results_nflx_only_new/iteration_2/results.pkl" ]; then
        echo "All 3 iterations completed. Test evaluation may be pending."
    fi
else
    ETIME=$(ps -p "$PID" -o etime= | tr -d ' ')
    echo "STATUS: RUNNING (PID: $PID, elapsed: $ETIME)"
    echo ""

    # Iteration progress
    IT0_DONE=$(grep -c "Iteration 0 completed" "$LOG" 2>/dev/null || echo 0)
    IT1_DONE=$(grep -c "Iteration 1 completed" "$LOG" 2>/dev/null || echo 0)
    IT2_DONE=$(grep -c "Iteration 2 completed" "$LOG" 2>/dev/null || echo 0)

    echo "Iterations: 0=$([ "$IT0_DONE" -gt 0 ] && echo 'DONE' || echo 'pending')  1=$([ "$IT1_DONE" -gt 0 ] && echo 'DONE' || echo 'pending')  2=$([ "$IT2_DONE" -gt 0 ] && echo 'DONE' || echo 'pending')"

    # Current sample being trained
    CUR_SAMPLE=$(grep "Training sample" "$LOG" | tail -1)
    CUR_EPISODE=$(grep "Episode" "$LOG" | tail -1 | grep -oP 'Episode \d+/\d+' | tail -1)

    echo "Current: $CUR_SAMPLE | $CUR_EPISODE"
    echo ""
    echo "Last 3 lines:"
    tail -3 "$LOG"
fi
echo "=========================================="
