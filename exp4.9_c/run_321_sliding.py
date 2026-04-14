#!/usr/bin/env python3
"""Exp4.9_c: 321方案并行执行器 - 训练3年|验证2年|测试1年"""
import os, sys, argparse, subprocess, datetime, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("/home/wangmeiyi/AuctionNet/lesr")
SCRIPT_DIR = ROOT / "exp4.9_c"
PYTHON = "/home/wangmeiyi/miniconda3/envs/lesr/bin/python"
NUM_GPUS = 4

def check_done(idx, test_year):
    return (ROOT / f"exp4.9_c/result_321_SW{idx:02d}_test{test_year}" / "test_set_results.pkl").exists()

def run_window(idx, test_year, gpu_id):
    cfg = SCRIPT_DIR / f"config_321_SW{idx:02d}.yaml"
    log = SCRIPT_DIR / "logs" / f"321_SW{idx:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cmd = [PYTHON, str(SCRIPT_DIR / "run_window.py"), '--config', str(cfg)]
    start = time.time()
    try:
        with open(log, 'w') as f:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=7200, env=env)
        elapsed = time.time() - start
        return idx, test_year, gpu_id, "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})", elapsed/60
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
    print(f"  Exp4.9_c: 321 Sliding Window ({NUM_GPUS} GPUs)")
    print(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Design: Train 3yr | Val 2yr | Test 1yr")
    print("=" * 70)

    tasks = []
    for idx in range(args.start, args.end + 1):
        ty = 2014 + idx
        if args.skip_existing and check_done(idx, ty):
            print(f"  321_SW{idx:02d} (test {ty}): skip")
            continue
        tasks.append((idx, ty))

    if not tasks:
        print("Nothing to run!"); return
    print(f"\n  {len(tasks)} windows, {NUM_GPUS} concurrent\n")

    results = []
    for bs in range(0, len(tasks), NUM_GPUS):
        batch = tasks[bs:bs + NUM_GPUS]
        print(f"\n--- Batch {bs//NUM_GPUS+1}: {', '.join(f'321_SW{t[0]:02d}' for t in batch)} ---")
        with ProcessPoolExecutor(max_workers=len(batch)) as ex:
            futs = {}
            for i, (idx, ty) in enumerate(batch):
                gpu = i % NUM_GPUS
                print(f"  321_SW{idx:02d} (test {ty}) -> GPU {gpu}")
                futs[ex.submit(run_window, idx, ty, gpu)] = idx
            for fut in as_completed(futs):
                idx, ty, gpu, st, el = fut.result()
                print(f"  [321_SW{idx:02d}] GPU{gpu} {st} ({el:.1f} min)")
                results.append((idx, ty, st))

    print(f"\n{'='*70}\n  Summary\n{'='*70}")
    for idx, ty, st in sorted(results):
        print(f"  321_SW{idx:02d} (test {ty}): {st}")

if __name__ == '__main__':
    main()
