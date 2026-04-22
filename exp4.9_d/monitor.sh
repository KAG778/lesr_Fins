#!/bin/bash
# exp4.9_d 监控脚本
LOG_DIR="/home/wangmeiyi/AuctionNet/lesr/exp4.9_d/logs"
REPORT="/home/wangmeiyi/AuctionNet/lesr/exp4.9_d/monitor_report.txt"
cd /home/wangmeiyi/AuctionNet/lesr/exp4.9_d

echo "========================================" > "$REPORT"
echo "exp4.9_d 监控 $(date '+%Y-%m-%d %H:%M:%S')" >> "$REPORT"
echo "========================================" >> "$REPORT"

running=$(ps aux | grep "run_window.py --config config_W" | grep -v grep | wc -l)
echo "运行中: ${running}/10" >> "$REPORT"
echo "" >> "$REPORT"

for w in W1 W2 W3 W4 W5 W6 W7 W8 W9 W10; do
  log=$(ls -t ${LOG_DIR}/${w}_2026*.log 2>/dev/null | head -1)
  [ -z "$log" ] && { echo "${w}: 无日志" >> "$REPORT"; continue; }

  done_flag=$(grep -c "Done!" "$log" 2>/dev/null)
  last_ep=$(grep -oP "Episode \d+/\d+" "$log" | tail -1)
  last_trades=$(grep -oP "Trades: \d+" "$log" | tail -1)
  iteration=$(grep -oP "Iteration \d+" "$log" | tail -1)
  sample=$(grep -oP "Training sample \d+/\d+" "$log" | tail -1)
  sharpe=$(grep -oP "\[.+?\] Sharpe: .+" "$log" | tail -1)
  error_count=$(grep -c "Error\|Traceback\|failed" "$log" 2>/dev/null)

  if [ "$done_flag" -gt 0 ]; then
    status="✓ 已完成"
  elif [ -n "$last_ep" ]; then
    status="训练 | ${iteration:-It0} | ${sample:---} | ${last_ep} | ${last_trades:---}"
  else
    current=$(grep -oP "\[Init\] Round \d+" "$log" | tail -1)
    valid=$(grep -c "validated" "$log" 2>/dev/null)
    status="Init | ${current:---} | valid=${valid}"
  fi

  echo "${w}: ${status}" >> "$REPORT"
  [ -n "$sharpe" ] && echo "  → ${sharpe}" >> "$REPORT"
done

echo "" >> "$REPORT"
echo "已完成:" >> "$REPORT"
for w in W1 W2 W3 W4 W5 W6 W7 W8 W9 W10; do
  dir=$(ls -d exp4.9_d/result_${w}_test* 2>/dev/null | head -1)
  [ -n "$dir" ] && [ -f "${dir}/test_set_results.pkl" ] && echo "  ${w}: ✓" >> "$REPORT"
done
echo "========================================" >> "$REPORT"

cat "$REPORT"
