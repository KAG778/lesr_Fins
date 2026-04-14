#!/usr/bin/env python3
"""每5分钟检查一次实验进度，写入日志"""
import time
import subprocess
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("/home/wangmeiyi/AuctionNet/lesr/exp4.9_c/logs/progress_monitor.log")
INTERVAL = 300  # 5 minutes

def check():
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result = subprocess.run(
        ["bash", "/home/wangmeiyi/AuctionNet/lesr/exp4.9_c/check_progress.sh"],
        capture_output=True, text=True
    )
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{ts}]\n")
        f.write(result.stdout)
        f.write("\n")
    print(result.stdout)

if __name__ == '__main__':
    print(f"Progress monitor started, interval={INTERVAL}s, log={LOG_FILE}")
    while True:
        check()
        time.sleep(INTERVAL)
