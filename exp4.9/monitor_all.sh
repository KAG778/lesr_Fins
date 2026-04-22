#!/bin/bash
# Monitor all 10 exp4.9 windows
cd /home/wangmeiyi/AuctionNet/lesr

echo "=== Exp4.9 Window Status ==="
for i in $(seq 1 10); do
    LOG="exp4.9/logs/result_W${i}*.log"
    # Find the actual log file
    LOGFILE=$(ls exp4.9/logs/result_W${i}_test*.log 2>/dev/null | head -1)
    if [ -z "$LOGFILE" ]; then
        LOGFILE="exp4.9/logs/W${i}.log"
    fi

    if [ -f "$LOGFILE" ]; then
        # Check progress
        LAST_EP=$(grep -oP 'Episode \K\d+' "$LOGFILE" 2>/dev/null | tail -1)
        LAST_ITER=$(grep -oP 'Iteration \K\d+' "$LOGFILE" 2>/dev/null | tail -1)
        SAMPLE=$(grep -oP 'Sample \d+.*validated' "$LOGFILE" 2>/dev/null | tail -1)
        DONE=$(grep -c "Done!" "$LOGFILE" 2>/dev/null)
        
        if [ "$DONE" -gt 0 ]; then
            STATUS="✅ 完成"
        elif [ -n "$LAST_EP" ]; then
            STATUS="🔄 It${LAST_ITER:-?} Ep${LAST_EP}"
        elif [ -n "$SAMPLE" ]; then
            STATUS="🔄 采样中"
        else
            STATUS="⏳ 启动中"
        fi
    else
        STATUS="❓ 未启动"
    fi
    echo "  W${i}: $STATUS"
done
echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
