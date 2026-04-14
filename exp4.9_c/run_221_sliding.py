#!/usr/bin/env python3
"""
Exp4.9_c: 221方案滑动窗口并行执行器
训练2年 | 验证2年 | 测试1年, 每年滑动, 共11窗口
4 GPU并行, 每窗口绑1张GPU
"""
import os, sys, argparse, subprocess, datetime, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("/home/wangmeiyi/AuctionNet/lesr")
SCRIPT_DIR = ROOT / "exp4.9_c"
PYTHON = "/home/wangmeiyi/miniconda3/envs/lesr/bin/python"
NUM_GPUS = 4
TOTAL_WINDOWS = 11  # test 2014-2024


def check_done(idx, test_year):
    return (ROOT / f"exp4.9_c/result_221_SW{idx:02d}_test{test_year}" / "test_set_results.pkl").exists()


def run_window(idx, test_year, gpu_id):
    config_file = SCRIPT_DIR / f"config_221_SW{idx:02d}.yaml"
    log_file = SCRIPT_DIR / "logs" / f"221_SW{idx:02d}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [PYTHON, str(SCRIPT_DIR / "run_window.py"), '--config', str(config_file)]

    start = time.time()
    try:
        with open(log_file, 'w') as f:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                                  timeout=7200, env=env)
        elapsed = time.time() - start
        ok = proc.returncode == 0
        return idx, test_year, gpu_id, "OK" if ok else f"FAIL(rc={proc.returncode})", elapsed/60
    except subprocess.TimeoutExpired:
        return idx, test_year, gpu_id, "TIMEOUT", 120
    except Exception as e:
        return idx, test_year, gpu_id, f"ERROR({e})", 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int,default=11)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Exp4.9_c: 221 Sliding Window ({NUM_GPUS} GPUs)")
    print(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Design: Train 2yr | Val 2yr | Test 1yr")
    print(f"  Windows: 221_SW{args.start:02d} ~ 221_SW{args.end:02d}")
    print("=" * 70)

    tasks = []
    for idx in range(args.start, args.end + 1):
        test_year = 2013 + idx
        if args.skip_existing and check_done(idx, test_year):
            print(f"  221_SW{idx:02d} (test {test_year}): skip (done)")
            continue
        tasks.append((idx, test_year))

    if not tasks:
        print("Nothing to run!"); return

    print(f"\n  {len(tasks)} windows, {NUM_GPUS} concurrent\n")

    results = []
    for batch_start in range(0, len(tasks), NUM_GPUS):
        batch = tasks[batch_start:batch_start + NUM_GPUS]
        print(f"\n--- Batch {batch_start//NUM_GPUS+1}: "
              f"{', '.join(f'221_SW{t[0]:02d}' for t in batch)} ---")

        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = {}
            for i, (idx, test_year) in enumerate(batch):
                gpu_id = i % NUM_GPUS
                print(f"  221_SW{idx:02d} (test {test_year}) -> GPU {gpu_id}")
                futures[executor.submit(run_window, idx, test_year, gpu_id)] = idx

            for future in as_completed(futures):
                idx, test_year, gpu_id, status, elapsed = future.result()
                print(f"  [221_SW{idx:02d}] GPU{gpu_id} {status} ({elapsed:.1f} min)")
                results.append((idx, test_year, status))

    print(f"\n{'='*70}")
    print("  Execution Summary")
    print(f"{'='*70}")
    for idx, test_year, status in sorted(results):
        print(f"  221_SW{idx:02d} (test {test_year}): {status}")


if __name__ == '__main__':
    main()
