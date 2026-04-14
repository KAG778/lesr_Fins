#!/bin/bash
echo "================================================================"
echo "  221滑动窗口实验进度  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  设计: 训练2年 | 验证2年 | 测试1年"
echo "================================================================"

DONE=0; RUNNING=0; WAITING=0

for i in $(seq 1 11); do
    YEAR=$((2013 + i))
    IDX=$(printf "%02d" $i)
    LOG="/home/wangmeiyi/AuctionNet/lesr/exp4.9_c/logs/221_SW${IDX}.log"
    RES="/home/wangmeiyi/AuctionNet/lesr/exp4.9_c/result_221_SW${IDX}_test${YEAR}"

    if [ -f "${RES}/test_set_results.pkl" ]; then
        echo "  221_SW${IDX} (test ${YEAR}): ✅ 完成"
        DONE=$((DONE + 1))
    elif [ -f "$LOG" ]; then
        LAST_ITER=$(grep -oP "Iteration \d+" "$LOG" | tail -1 | grep -oP "\d+")
        LAST_EP=$(grep -oP "Episode \d+/\d+" "$LOG" | tail -1)
        LAST_LLM=$(grep "calling LLM" "$LOG" | tail -1 | grep -oP "R\d+ S\d+" | tail -1)
        if [ -n "$LAST_EP" ]; then
            echo "  221_SW${IDX} (test ${YEAR}): 🔄 It${LAST_ITER:-0} ${LAST_EP}"
        elif [ -n "$LAST_LLM" ]; then
            echo "  221_SW${IDX} (test ${YEAR}): 🔄 It${LAST_ITER:-0} LLM ${LAST_LLM}"
        else
            echo "  221_SW${IDX} (test ${YEAR}): 🔄 启动中"
        fi
        RUNNING=$((RUNNING + 1))
    else
        echo "  221_SW${IDX} (test ${YEAR}): ⬜ 等待"
        WAITING=$((WAITING + 1))
    fi
done

echo ""
echo "  完成: ${DONE}/11 | 运行中: ${RUNNING} | 等待: ${WAITING}"
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "无GPU"
