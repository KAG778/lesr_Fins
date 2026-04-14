#!/usr/bin/env python3
"""
Exp4.9_c: Sliding Window Experiment Runner
滑动窗口: 训练3年 | 验证1年 | 测试1年, 每年滑动一次, 共10个窗口
SW01~SW10: test 2015 ~ 2024

Usage:
  python exp4.9_c/run_sliding_windows.py                    # 运行全部10个窗口
  python exp4.9_c/run_sliding_windows.py --start 1 --end 3  # 只运行 SW01~SW03
  python exp4.9_c/run_sliding_windows.py --skip-existing     # 跳过已完成的窗口
"""
import os
import sys
import argparse
import subprocess
import datetime
import time
from pathlib import Path

ROOT = Path("/home/wangmeiyi/AuctionNet/lesr")
SCRIPT_DIR = ROOT / "exp4.9_c"
PYTHON = "/home/wangmeiyi/miniconda3/envs/lesr/bin/python"


def check_result_exists(result_dir):
    """检查窗口是否已完成 (有 test_set_results.pkl)"""
    return (Path(result_dir) / 'test_set_results.pkl').exists()


def run_window(idx, test_year):
    config_file = SCRIPT_DIR / f"config_SW{idx:02d}.yaml"
    result_dir = ROOT / f"exp4.9_c/result_SW{idx:02d}_test{test_year}"

    if not config_file.exists():
        print(f"[SW{idx:02d}] Config not found: {config_file}")
        return False

    log_file = SCRIPT_DIR / "logs" / f"sliding_SW{idx:02d}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  SW{idx:02d}: test {test_year} | config {config_file.name}")
    print(f"  Log: {log_file}")
    print(f"  Result: {result_dir}")
    print(f"{'='*70}")

    cmd = [
        PYTHON, str(SCRIPT_DIR / "run_window.py"),
        '--config', str(config_file),
    ]

    start = time.time()
    try:
        with open(log_file, 'w') as log_f:
            proc = subprocess.run(
                cmd, cwd=str(ROOT),
                stdout=log_f, stderr=subprocess.STDOUT,
                timeout=7200  # 2 hour timeout per window
            )
        elapsed = time.time() - start
        ok = proc.returncode == 0
        status = "SUCCESS" if ok else f"FAILED (rc={proc.returncode})"
        print(f"  [{status}] in {elapsed/60:.1f} min")
        return ok
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] after 2 hours")
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def collect_results():
    """汇总所有滑动窗口结果"""
    import pickle

    print(f"\n\n{'='*80}")
    print("  滑动窗口实验汇总 (Sliding Window Summary)")
    print(f"{'='*80}")

    header = f"{'窗口':<8} {'测试年':<8} "
    for t in ['TSLA', 'NFLX', 'AMZN', 'MSFT']:
        header += f"{'LESR_'+t:<12} {'Base_'+t:<12} "
    header += "  Win Rate"
    print(header)
    print("-" * len(header))

    all_results = {}
    total_win = 0
    total_compare = 0

    for idx in range(1, 11):
        test_year = 2014 + idx
        result_dir = ROOT / f"exp4.9_c/result_SW{idx:02d}_test{test_year}"
        pkl_file = result_dir / 'test_set_results.pkl'

        if not pkl_file.exists():
            line = f"SW{idx:02d}    {test_year}    (未完成)"
            print(line)
            continue

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        line = f"SW{idx:02d}    {test_year}   "
        window_wins = 0
        window_total = 0
        for t in ['TSLA', 'NFLX', 'AMZN', 'MSFT']:
            if t in data and data[t].get('error') is None:
                ls = data[t]['lesr_test']['sharpe']
                bs = data[t]['baseline_test']['sharpe']
                line += f"{ls:>8.3f}      {bs:>8.3f}    "
                if ls > bs:
                    window_wins += 1
                window_total += 1
            else:
                line += f"{'N/A':>8}      {'N/A':>8}    "
        if window_total > 0:
            wr = window_wins / window_total * 100
            line += f"  {window_wins}/{window_total} ({wr:.0f}%)"
            total_win += window_wins
            total_compare += window_total
        print(line)
        all_results[f'SW{idx:02d}'] = data

    print("-" * len(header))
    if total_compare > 0:
        print(f"\n总体 LESR 胜率: {total_win}/{total_compare} ({total_win/total_compare*100:.1f}%)")

    # Save summary
    summary_file = SCRIPT_DIR / "sliding_window_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Sliding Window Experiment Summary\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Window design: train 3yr | val 1yr | test 1yr, slide 1yr\n")
        f.write(f"Total LESR win rate: {total_win}/{total_compare}\n")
    print(f"\nSummary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Run sliding window experiments')
    parser.add_argument('--start', type=int, default=1, help='Start window index (1-10)')
    parser.add_argument('--end', type=int, default=10, help='End window index (1-10)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip windows with results')
    args = parser.parse_args()

    print("=" * 70)
    print("  Exp4.9_c: Sliding Window Experiment")
    print(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Windows: SW{args.start:02d} ~ SW{args.end:02d}")
    print("=" * 70)

    results = {}
    for idx in range(args.start, args.end + 1):
        test_year = 2014 + idx
        result_dir = ROOT / f"exp4.9_c/result_SW{idx:02d}_test{test_year}"

        if args.skip_existing and check_result_exists(result_dir):
            print(f"\n[SW{idx:02d}] Skipping (already done)")
            results[idx] = True
            continue

        ok = run_window(idx, test_year)
        results[idx] = ok

    # Summary
    print(f"\n{'='*70}")
    print("  Execution Summary")
    print(f"{'='*70}")
    for idx, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  SW{idx:02d}: {status}")

    # Collect results
    collect_results()


if __name__ == '__main__':
    main()
