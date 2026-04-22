#!/bin/bash
cd /home/wangmeiyi/AuctionNet/lesr/exp4.9_d
mkdir -p logs

echo "========================================"
echo "exp4.9_d: 启动 10 窗口评测"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

for w in W1 W2 W3 W4 W5 W6 W7 W8 W9 W10; do
  config="config_${w}.yaml"
  log_file="logs/${w}_$(date +%Y%m%d_%H%M%S).log"
  echo "启动 ${w}: ${config}"
  nohup python run_window.py --config ${config} > ${log_file} 2>&1 &
  echo "  PID: $!"
done

echo ""
echo "所有窗口已启动！"
echo "监控: bash monitor.sh"
