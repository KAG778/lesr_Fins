#!/usr/bin/env python3
"""修复版321: SW01-04, 4 GPU并行"""
import os, sys, subprocess, datetime, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path("/home/wangmeiyi/AuctionNet/lesr")
SD = ROOT / "exp4.9_c"
PY = "/home/wangmeiyi/miniconda3/envs/lesr/bin/python"

def run(idx, ty, gpu):
    cfg = SD / f"config_321_fixed_SW{idx:02d}.yaml"
    log = SD / "logs" / f"321_fixed_SW{idx:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cmd = [PY, str(SD / "run_window.py"), '--config', str(cfg)]
    t0 = time.time()
    try:
        with open(log, 'w') as f:
            p = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, timeout=7200, env=env)
        el = (time.time() - t0) / 60
        return idx, ty, gpu, "OK" if p.returncode == 0 else f"FAIL({p.returncode})", el
    except subprocess.TimeoutExpired:
        return idx, ty, gpu, "TIMEOUT", 120
    except Exception as e:
        return idx, ty, gpu, f"ERR({e})", 0

def main():
    print(f"=== 321 Fixed (no-random): {datetime.datetime.now().strftime('%H:%M:%S')} ===")
    tasks = [(1,2015),(2,2016),(3,2017),(4,2018)]
    results = []
    with ProcessPoolExecutor(4) as ex:
        futs = {}
        for i, (idx, ty) in enumerate(tasks):
            print(f"  SW{idx:02d} test {ty} -> GPU {i}")
            futs[ex.submit(run, idx, ty, i)] = idx
        for f in as_completed(futs):
            idx, ty, gpu, st, el = f.result()
            print(f"  [SW{idx:02d}] GPU{gpu} {st} ({el:.1f}m)")
            results.append((idx, st))
    print(f"\nSummary:")
    for idx, st in sorted(results):
        print(f"  321_fixed_SW{idx:02d}: {st}")

if __name__ == '__main__':
    main()
