#!/usr/bin/env python3
"""
Exp4.9_c: Sliding Window Parallel Runner
每个窗口绑定一张GPU，4窗口同时跑

Usage:
  python exp4.9_c/run_sliding_parallel.py                     # 并行运行全部
  python exp4.9_c/run_sliding_parallel.py --start 1 --end 8   # 运行 SW01~SW08
  python exp4.9_c/run_sliding_parallel.py --skip-existing      # 跳过已完成的
"""
import os
import sys
import argparse
import subprocess
import datetime
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("/home/wangmeiyi/AuctionNet/lesr")
SCRIPT_DIR = ROOT / "exp4.9_c"
PYTHON = "/home/wangmeiyi/miniconda3/envs/lesr/bin/python"
NUM_GPUS = 4


def check_result_exists(idx, test_year):
    result_dir = ROOT / f"exp4.9_c/result_SW{idx:02d}_test{test_year}"
    return (result_dir / 'test_set_results.pkl').exists()


def run_single_window(idx, test_year, gpu_id):
    """单个窗口：强制只使用指定GPU"""
    config_file = SCRIPT_DIR / f"config_SW{idx:02d}.yaml"
    log_file = SCRIPT_DIR / "logs" / f"sliding_SW{idx:02d}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # 注入 CUDA_VISIBLE_DEVICES 限制只用一张GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [PYTHON, str(SCRIPT_DIR / "run_window.py"),
           '--config', str(config_file)]

    start = time.time()
    try:
        with open(log_file, 'w') as log_f:
            proc = subprocess.run(
                cmd, cwd=str(ROOT),
                stdout=log_f, stderr=subprocess.STDOUT,
                timeout=7200, env=env
            )
        elapsed = time.time() - start
        ok = proc.returncode == 0
        status = "OK" if ok else f"FAIL(rc={proc.returncode})"
        return idx, test_year, gpu_id, status, elapsed / 60
    except subprocess.TimeoutExpired:
        return idx, test_year, gpu_id, "TIMEOUT", 120
    except Exception as e:
        return idx, test_year, gpu_id, f"ERROR({e})", 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Exp4.9_c: Parallel Sliding Window ({NUM_GPUS} GPUs)")
    print(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Windows: SW{args.start:02d} ~ SW{args.end:02d}")
    print(f"  Strategy: 1 window per GPU, {NUM_GPUS} windows concurrent")
    print("=" * 70)

    # Build task list
    tasks = []
    for idx in range(args.start, args.end + 1):
        test_year = 2014 + idx
        if args.skip_existing and check_result_exists(idx, test_year):
            print(f"  SW{idx:02d} (test {test_year}): skip (done)")
            continue
        tasks.append((idx, test_year))

    if not tasks:
        print("Nothing to run!")
        return

    print(f"\n  {len(tasks)} windows to run, {NUM_GPUS} at a time\n")

    results = []
    # 分批执行，每批 NUM_GPUS 个
    for batch_start in range(0, len(tasks), NUM_GPUS):
        batch = tasks[batch_start:batch_start + NUM_GPUS]
        print(f"\n--- Batch {batch_start // NUM_GPUS + 1}: "
              f"{', '.join(f'SW{t[0]:02d}' for t in batch)} ---")

        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {}
            for i, (idx, test_year) in enumerate(batch):
                gpu_id = i % NUM_GPUS
                print(f"  SW{idx:02d} (test {test_year}) -> GPU {gpu_id}")
                futures[executor.submit(run_single_window, idx, test_year, gpu_id)] = idx

            for future in as_completed(futures):
                idx, test_year, gpu_id, status, elapsed = future.result()
                print(f"  [SW{idx:02d}] GPU{gpu_id} {status} ({elapsed:.1f} min)")
                results.append((idx, test_year, status))

    # Summary
    print(f"\n{'='*70}")
    print("  Execution Summary")
    print(f"{'='*70}")
    for idx, test_year, status in sorted(results):
        print(f"  SW{idx:02d} (test {test_year}): {status}")

    # Run final summary
    print("\nGenerating summary report...")
    subprocess.run([PYTHON, str(SCRIPT_DIR / "sliding_summary.py")], cwd=str(ROOT))


if __name__ == '__main__':
    main()
