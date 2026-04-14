#!/bin/bash
# 滑动窗口实验进度检查
LOG_DIR="/home/wangmeiyi/AuctionNet/lesr/exp4.9_c/logs"
RESULT_DIR="/home/wangmeiyi/AuctionNet/lesr/exp4.9_c"
PYTHON="/home/wangmeiyi/miniconda3/envs/lesr/bin/python"

echo "================================================================"
echo "  滑动窗口实验进度  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

DONE=0
RUNNING=0
WAITING=0

for i in $(seq 1 10); do
    YEAR=$((2014 + i))
    IDX=$(printf "%02d" $i)
    LOG="${LOG_DIR}/sliding_SW${IDX}.log"
    RES="${RESULT_DIR}/result_SW${IDX}_test${YEAR}"

    if [ -f "${RES}/test_set_results.pkl" ]; then
        echo "  SW${IDX} (test ${YEAR}): ✅ 完成"
        DONE=$((DONE + 1))
    elif [ -f "$LOG" ]; then
        # 获取最新迭代
        LAST_ITER=$(grep -oP "Iteration \d+" "$LOG" | tail -1 | grep -oP "\d+")
        LAST_EP=$(grep -oP "Episode \d+/\d+" "$LOG" | tail -1)
        LAST_LLM=$(grep "calling LLM" "$LOG" | tail -1 | grep -oP "R\d+ S\d+" | tail -1)
        
        if [ -n "$LAST_EP" ]; then
            echo "  SW${IDX} (test ${YEAR}): 🔄 It${LAST_ITER:-0} ${LAST_EP}"
        elif [ -n "$LAST_LLM" ]; then
            echo "  SW${IDX} (test ${YEAR}): 🔄 It${LAST_ITER:-0} LLM ${LAST_LLM}"
        else
            echo "  SW${IDX} (test ${YEAR}): 🔄 启动中"
        fi
        RUNNING=$((RUNNING + 1))
    else
        echo "  SW${IDX} (test ${YEAR}): ⬜ 等待"
        WAITING=$((WAITING + 1))
    fi
done

echo ""
echo "  完成: ${DONE}/10 | 运行中: ${RUNNING} | 等待: ${WAITING}"
echo ""

# GPU
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "无GPU"
echo ""

# 最新日志尾部
LATEST_LOG=$(ls -t ${LOG_DIR}/sliding_SW*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "--- 最新活跃日志 (5行) ---"
    echo ">>> $(basename $LATEST_LOG)"
    grep -E "Episode|Iteration|Validated|Sharpe|ERROR" "$LATEST_LOG" | tail -5
fi
echo ""
