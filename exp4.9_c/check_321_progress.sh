#!/bin/bash
echo "================================================================"
echo "  321滑动窗口实验进度  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  设计: 训练3年 | 验证2年 | 测试1年"
echo "================================================================"
DONE=0; RUNNING=0; WAITING=0
for i in $(seq 1 10); do
    YEAR=$((2014 + i))
    IDX=$(printf "%02d" $i)
    LOG="/home/wangmeiyi/AuctionNet/lesr/exp4.9_c/logs/321_SW${IDX}.log"
    RES="/home/wangmeiyi/AuctionNet/lesr/exp4.9_c/result_321_SW${IDX}_test${YEAR}"
    if [ -f "${RES}/test_set_results.pkl" ]; then
        echo "  321_SW${IDX} (test ${YEAR}): ✅ 完成"
        DONE=$((DONE + 1))
    elif [ -f "$LOG" ]; then
        LI=$(grep -oP "Iteration \d+" "$LOG" | tail -1 | grep -oP "\d+")
        LE=$(grep -oP "Episode \d+/\d+" "$LOG" | tail -1)
        LL=$(grep "calling LLM" "$LOG" | tail -1 | grep -oP "R\d+ S\d+" | tail -1)
        if [ -n "$LE" ]; then echo "  321_SW${IDX} (test ${YEAR}): 🔄 It${LI:-0} ${LE}"
        elif [ -n "$LL" ]; then echo "  321_SW${IDX} (test ${YEAR}): 🔄 It${LI:-0} LLM ${LL}"
        else echo "  321_SW${IDX} (test ${YEAR}): 🔄 启动中"; fi
        RUNNING=$((RUNNING + 1))
    else
        echo "  321_SW${IDX} (test ${YEAR}): ⬜ 等待"
        WAITING=$((WAITING + 1))
    fi
done
echo ""
echo "  完成: ${DONE}/10 | 运行中: ${RUNNING} | 等待: ${WAITING}"
echo ""
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
