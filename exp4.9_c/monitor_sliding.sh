#!/bin/bash
# 监控滑动窗口实验进度
echo "=========================================="
echo "  滑动窗口实验进度监控"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

for i in $(seq 1 10); do
    YEAR=$((2014 + i))
    IDX=$(printf "%02d" $i)
    RESULT_DIR="exp4.9_c/result_SW${IDX}_test${YEAR}"
    LOG_FILE="exp4.9_c/logs/sliding_SW${IDX}.log"

    if [ -f "${RESULT_DIR}/test_set_results.pkl" ]; then
        echo "  SW${IDX} (test ${YEAR}): ✅ 完成"
    elif [ -f "${RESULT_DIR}/iteration_0/results.pkl" ]; then
        # 检查进展
        LAST_ITER=$(ls -d ${RESULT_DIR}/iteration_* 2>/dev/null | tail -1)
        LAST_ITER=$(basename $LAST_ITER 2>/dev/null)
        echo "  SW${IDX} (test ${YEAR}): 🔄 运行中 (${LAST_ITER})"
    elif [ -f "${LOG_FILE}" ]; then
        echo "  SW${IDX} (test ${YEAR}): ⏳ 日志存在"
    else
        echo "  SW${IDX} (test ${YEAR}): ⬜ 未开始"
    fi
done

echo ""
echo "--- GPU状态 ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "无GPU"

echo ""
echo "--- 最新日志 (最后10行) ---"
LATEST_LOG=$(ls -t exp4.9_c/logs/sliding_SW*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo ">>> $LATEST_LOG"
    tail -10 "$LATEST_LOG"
fi
