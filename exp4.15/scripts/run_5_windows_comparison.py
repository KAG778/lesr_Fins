#!/usr/bin/env python3
"""
批量运行 5 个窗口（W2-W6）的 LESR 和 Baseline DQN 对比实验

使用 W1_fixed_v6 的结构进行测试
"""
import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保工作目录正确
script_dir = Path(__file__).parent
project_dir = script_dir.parent
os.chdir(project_dir)

# 要测试的窗口
WINDOWS = ['W2', 'W3', 'W4', 'W5', 'W6']

# 实验配置
EXPERIMENT_TAG = "5_windows_comparison"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def run_lesr(window: str):
    """运行指定窗口的 LESR 实验"""
    logger.info(f"\n{'='*60}")
    logger.info(f"开始运行 {window} LESR 实验")
    logger.info(f"{'='*60}")

    config_path = f"configs/config_{window}.yaml"
    log_path = f"logs/{window}_lesr_{TIMESTAMP}.log"

    cmd = [
        sys.executable,
        "scripts/main_simple.py",
        "--config", config_path
    ]

    logger.info(f"命令: {' '.join(cmd)}")

    try:
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

            # 等待进程完成
            return_code = process.wait()

        if return_code == 0:
            logger.info(f"✓ {window} LESR 实验完成")
            return True
        else:
            logger.error(f"✗ {window} LESR 实验失败 (返回码: {return_code})")
            return False

    except Exception as e:
        logger.error(f"✗ {window} LESR 实验出错: {e}")
        return False


def run_baseline(window: str):
    """运行指定窗口的 Baseline DQN 实验"""
    logger.info(f"\n{'='*60}")
    logger.info(f"开始运行 {window} Baseline DQN 实验")
    logger.info(f"{'='*60}")

    config_path = f"configs/config_{window}.yaml"
    log_path = f"logs/{window}_baseline_{TIMESTAMP}.log"

    cmd = [
        sys.executable,
        "scripts/run_baseline.py",
        "--config", config_path
    ]

    logger.info(f"命令: {' '.join(cmd)}")

    try:
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

            # 等待进程完成
            return_code = process.wait()

        if return_code == 0:
            logger.info(f"✓ {window} Baseline 实验完成")
            return True
        else:
            logger.error(f"✗ {window} Baseline 实验失败 (返回码: {return_code})")
            return False

    except Exception as e:
        logger.error(f"✗ {window} Baseline 实验出错: {e}")
        return False


def run_window_comparison(window: str):
    """运行单个窗口的完整对比实验（LESR + Baseline）"""
    logger.info(f"\n{'#'*60}")
    logger.info(f"# 开始窗口 {window} 的完整对比实验")
    logger.info(f"# {TIMESTAMP}")
    logger.info(f"{'#'*60}")

    results = {
        'window': window,
        'lesr_success': False,
        'baseline_success': False,
        'lesr_log': f"logs/{window}_lesr_{TIMESTAMP}.log",
        'baseline_log': f"logs/{window}_baseline_{TIMESTAMP}.log"
    }

    # 运行 LESR
    logger.info(f"\n[1/2] 运行 LESR...")
    results['lesr_success'] = run_lesr(window)

    # 运行 Baseline
    logger.info(f"\n[2/2] 运行 Baseline DQN...")
    results['baseline_success'] = run_baseline(window)

    # 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info(f"窗口 {window} 实验总结:")
    logger.info(f"  LESR: {'✓ 成功' if results['lesr_success'] else '✗ 失败'}")
    logger.info(f"  Baseline: {'✓ 成功' if results['baseline_success'] else '✗ 失败'}")
    logger.info(f"  LESR 日志: {results['lesr_log']}")
    logger.info(f"  Baseline 日志: {results['baseline_log']}")
    logger.info(f"{'='*60}")

    return results


def main():
    """主函数：运行所有窗口的对比实验"""
    logger.info(f"\n{'#'*60}")
    logger.info(f"# 5 窗口 LESR vs Baseline DQN 对比实验")
    logger.info(f"# 使用 W1_fixed_v6 结构")
    logger.info(f"# 窗口: {', '.join(WINDOWS)}")
    logger.info(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*60}")

    all_results = {}

    # 按顺序运行每个窗口
    for i, window in enumerate(WINDOWS, 1):
        logger.info(f"\n\n{'='*60}")
        logger.info(f"进度: [{i}/{len(WINDOWS)}] 处理窗口 {window}")
        logger.info(f"{'='*60}")

        result = run_window_comparison(window)
        all_results[window] = result

    # 最终汇总
    logger.info(f"\n\n{'#'*60}")
    logger.info(f"# 所有实验完成!")
    logger.info(f"# 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'#'*60}")

    logger.info(f"\n{'='*60}")
    logger.info("最终汇总:")
    for window, result in all_results.items():
        lesr_status = "✓" if result['lesr_success'] else "✗"
        baseline_status = "✓" if result['baseline_success'] else "✗"
        logger.info(f"  {window}: LESR {lesr_status} | Baseline {baseline_status}")
    logger.info(f"{'='*60}")

    # 统计成功率
    lesr_success_count = sum(1 for r in all_results.values() if r['lesr_success'])
    baseline_success_count = sum(1 for r in all_results.values() if r['baseline_success'])

    logger.info(f"\n成功率统计:")
    logger.info(f"  LESR: {lesr_success_count}/{len(WINDOWS)} ({lesr_success_count/len(WINDOWS)*100:.1f}%)")
    logger.info(f"  Baseline: {baseline_success_count}/{len(WINDOWS)} ({baseline_success_count/len(WINDOWS)*100:.1f}%)")


if __name__ == '__main__':
    main()
