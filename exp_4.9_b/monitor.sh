#!/bin/bash
# exp_4.9_b 定时监控脚本 - 每 10 分钟运行一次
LOG_DIR="/home/wangmeiyi/AuctionNet/lesr/exp_4.9_b/logs"
REPORT_FILE="/home/wangmeiyi/AuctionNet/lesr/exp_4.9_b/monitor_report.txt"
cd /home/wangmeiyi/AuctionNet/lesr/exp_4.9_b

echo "========================================" > "$REPORT_FILE"
echo "exp_4.9_b 监控报告 $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT_FILE"
echo "========================================" >> "$REPORT_FILE"

running=$(ps aux | grep "run_window.py --config config_W" | grep -v grep | wc -l)
echo "运行中进程: ${running}/10" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

for w in W1 W2 W3 W4 W5 W6 W7 W8 W9 W10; do
  log=$(ls -t ${LOG_DIR}/${w}_2026*.log 2>/dev/null | head -1)
  if [ -z "$log" ]; then
    echo "${w}: 无日志文件" >> "$REPORT_FILE"
    continue
  fi

  init_count=$(grep -c "validated:" "$log" 2>/dev/null)
  current_round=$(grep -oP "\[Init\] Round \d+" "$log" | tail -1)
  last_episode=$(grep -oP "Episode \d+/\d+" "$log" | tail -1)
  last_trades=$(grep -oP "Trades: \d+" "$log" | tail -1)
  iteration=$(grep -oP "Iteration \d+" "$log" | tail -1)
  sample=$(grep -oP "Training sample \d+/\d+" "$log" | tail -1)
  sharpe=$(grep -oP "\[.+?\] Sharpe: .+" "$log" | tail -1)
  done_flag=$(grep -c "Done!" "$log" 2>/dev/null)

  if [ "$done_flag" -gt 0 ]; then
    status="✓ 已完成"
  elif [ -n "$last_episode" ]; then
    status="训练中 | ${iteration:-It0} | ${sample:---} | ${last_episode} | ${last_trades:---}"
  elif [ -n "$current_round" ]; then
    status="Init采样 | ${current_round} | 已验证${init_count}个"
  else
    status="启动中..."
  fi

  echo "${w}: ${status}" >> "$REPORT_FILE"
  if [ -n "$sharpe" ]; then
    echo "  → ${sharpe}" >> "$REPORT_FILE"
  fi
done

echo "" >> "$REPORT_FILE"
echo "=== 已完成窗口 ===" >> "$REPORT_FILE"
for w in W1 W2 W3 W4 W5 W6 W7 W8 W9 W10; do
  result_dir=$(ls -d result_${w}_test* 2>/dev/null | head -1)
  if [ -n "$result_dir" ] && [ -f "${result_dir}/test_set_results.pkl" ]; then
    echo "${w}: 测试集评估完成" >> "$REPORT_FILE"
  fi
done
echo "========================================" >> "$REPORT_FILE"

cat "$REPORT_FILE"
