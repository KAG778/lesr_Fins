#!/bin/bash
# exp_4.9_b: 启动所有 10 个窗口的评测
# 每个窗口在后台独立运行，日志输出到 logs/ 目录

cd /home/wangmeiyi/AuctionNet/lesr/exp_4.9_b

# 确保 logs 目录存在
mkdir -p logs

echo "========================================"
echo "exp_4.9_b: 启动 10 个窗口评测"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 每个窗口用 nohup 后台运行
declare -A WINDOWS=(
    ["W1"]="config_W1.yaml"
    ["W2"]="config_W2.yaml"
    ["W3"]="config_W3.yaml"
    ["W4"]="config_W4.yaml"
    ["W5"]="config_W5.yaml"
    ["W6"]="config_W6.yaml"
    ["W7"]="config_W7.yaml"
    ["W8"]="config_W8.yaml"
    ["W9"]="config_W9.yaml"
    ["W10"]="config_W10.yaml"
)

for window in W1 W2 W3 W4 W5 W6 W7 W8 W9 W10; do
    config=${WINDOWS[$window]}
    log_file="logs/${window}_$(date +%Y%m%d_%H%M%S).log"

    echo "启动 ${window}: ${config} -> ${log_file}"
    nohup python run_window.py --config ${config} > ${log_file} 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "所有窗口已启动！"
echo ""
echo "监控命令:"
echo "  tail -f logs/W1_*.log    # 查看单个窗口"
echo "  ls -la logs/              # 查看所有日志文件"
echo "  ps aux | grep run_window  # 查看运行进程"
echo ""
echo "结果目录:"
echo "  ls result_W*_test*/"
