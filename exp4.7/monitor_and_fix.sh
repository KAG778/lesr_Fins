#!/bin/bash
# Exp4.7 自动监控和修复脚本

cd /home/wangmeiyi/AuctionNet/lesr

LOG_FILE="exp4.7/logs/monitor.log"
mkdir -p exp4.7/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_processes() {
    LESR_COUNT=$(ps aux | grep "main_simple.py" | grep -v grep | wc -l)
    BASELINE_COUNT=$(ps aux | grep "train_baseline.py" | grep -v grep | wc -l)

    if [ "$LESR_COUNT" -eq 0 ]; then
        log "⚠️  LESR进程已停止，检查原因..."
        tail -50 exp4.7/logs/run.log | tail -20
    fi

    if [ "$BASELINE_COUNT" -eq 0 ]; then
        log "⚠️  基线进程已停止，检查原因..."
        tail -50 exp4.7/logs/baseline.log | tail -20
    fi

    echo "LESR: $LESR_COUNT | Baseline: $BASELINE_COUNT"
}

check_errors() {
    # 检查是否有严重错误
    if grep -q "MemoryError\|Killed\|CUDA out of memory" exp4.7/logs/run.log 2>/dev/null; then
        log "🔴 检测到内存错误，可能需要减少batch size"
        return 1
    fi

    if grep -q "Connection\|Timeout\|Network" exp4.7/logs/run.log 2>/dev/null; then
        log "🟡 检测到网络问题，可能会自动重试"
        return 2
    fi

    return 0
}

restart_if_needed() {
    # 检查进程是否意外停止
    LESR_COUNT=$(ps aux | grep "main_simple.py" | grep -v grep | wc -l)

    if [ "$LESR_COUNT" -eq 0 ]; then
        # 检查是否正常完成
        if grep -q "优化完成\|Best strategy" exp4.7/logs/run.log 2>/dev/null; then
            log "✅ LESR训练已正常完成"
            return 0
        fi

        # 检查是否是崩溃
        if grep -q "Traceback\|Exception" exp4.7/logs/run.log 2>/dev/null; then
            log "🔴 LESR进程崩溃，尝试重启..."
            source /home/wangmeiyi/miniconda3/etc/profile.d/conda.sh
            conda activate finsaber
            export PYTHONPATH=/home/wangmeiyi/AuctionNet/lesr/FINSABER:/home/wangmeiyi/AuctionNet/lesr/exp4.7:/home/wangmeiyi/AuctionNet/lesr:$PYTHONPATH

            nohup python exp4.7/main_simple.py > exp4.7/logs/run_restart.log 2>&1 &
            log "🔄 LESR已重启"
        fi
    fi
}

log "=== 启动监控 ==="

while true; do
    check_processes
    check_errors
    restart_if_needed

    # 每2分钟检查一次
    sleep 120
done
