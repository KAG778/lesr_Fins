#!/usr/bin/env python3
"""
监控 5 窗口对比实验的进度
"""
import os
import re
from datetime import datetime
from pathlib import Path

def parse_log_file(log_path):
    """解析日志文件，提取关键信息"""
    if not os.path.exists(log_path):
        return None

    info = {
        'exists': True,
        'completed': False,
        'current_iteration': None,
        'best_sharpe': None,
        'test_results': {},
        'error': None
    }

    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # 检查是否完成
        if '优化完成!' in content or 'Summary:' in content:
            info['completed'] = True

        # 查找当前迭代
        iterations = re.findall(r'=== Iteration (\d+) ===', content)
        if iterations:
            info['current_iteration'] = max(int(it) for it in iterations)

        # 查找最佳 Sharpe
        best_match = re.search(r'Best: It(\d+), Sharpe=([\d.]+)', content)
        if best_match:
            info['best_sharpe'] = float(best_match.group(2))

        # 查找测试结果
        test_matches = re.findall(r'\[(\w+)\] (?:Val )?Test Sharpe=([\d.-]+)', content)
        for ticker, sharpe in test_matches:
            if ticker not in info['test_results']:
                info['test_results'][ticker] = []
            info['test_results'][ticker].append(float(sharpe))

        # 查找错误
        error_matches = re.findall(r'ERROR - (.+)', content)
        if error_matches:
            info['error'] = error_matches[-1][:100]  # 只保留最后一个错误的前100字符

    except Exception as e:
        info['error'] = f"读取错误: {str(e)}"

    return info


def monitor_experiment():
    """监控实验进度"""
    print(f"\n{'='*70}")
    print(f"5 窗口对比实验监控")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    windows = ['W2', 'W3', 'W4', 'W5', 'W6']

    # 查找最新的实验日志
    log_dir = Path('/home/wangmeiyi/AuctionNet/lesr/exp4.15/logs')
    lesl_logs = {}
    baseline_logs = {}

    for window in windows:
        # 查找 LESR 日志
        lesl_pattern = f"{window}_lesr_*.log"
        lesl_files = sorted(log_dir.glob(lesl_pattern), reverse=True)
        if lesl_files:
            lesl_logs[window] = lesl_files[0]

        # 查找 Baseline 日志
        baseline_pattern = f"{window}_baseline_*.log"
        baseline_files = sorted(log_dir.glob(baseline_pattern), reverse=True)
        if baseline_files:
            baseline_logs[window] = baseline_files[0]

    # 显示 LESR 状态
    print("【LESR 实验状态】")
    print(f"{'窗口':<8} {'状态':<12} {'迭代':<8} {'最佳 Sharpe':<12} {'最新测试 Sharpe'}")
    print("-" * 70)

    for window in windows:
        if window in lesl_logs:
            info = parse_log_file(lesl_logs[window])
            if info:
                status = "✅ 完成" if info['completed'] else "🔄 运行中"
                iteration = info['current_iteration'] if info['current_iteration'] else "N/A"
                best_sharpe = f"{info['best_sharpe']:.3f}" if info['best_sharpe'] else "N/A"

                # 获取最新的测试 Sharpe
                latest_test = "N/A"
                if info['test_results']:
                    for ticker, sharpes in info['test_results'].items():
                        if sharpes:
                            latest_test = f"{sharpes[-1]:.3f}"
                            break

                print(f"{window:<8} {status:<12} {iteration:<8} {best_sharpe:<12} {latest_test}")

                if info['error']:
                    print(f"  ⚠️  错误: {info['error']}")
            else:
                print(f"{window:<8} ❌ 读取失败")
        else:
            print(f"{window:<8} ⏳ 未开始")

    # 显示 Baseline 状态
    print(f"\n{'='*70}")
    print("【Baseline DQN 实验状态】")
    print(f"{'窗口':<8} {'状态':<12} {'平均测试 Sharpe':<15}")
    print("-" * 70)

    for window in windows:
        if window in baseline_logs:
            info = parse_log_file(baseline_logs[window])
            if info:
                status = "✅ 完成" if info['completed'] else "🔄 运行中"
                avg_sharpe = "N/A"
                if info['test_results']:
                    all_sharpes = []
                    for sharpes in info['test_results'].values():
                        all_sharpes.extend(sharpes)
                    if all_sharpes:
                        avg_sharpe = f"{sum(all_sharpes)/len(all_sharpes):.3f}"

                print(f"{window:<8} {status:<12} {avg_sharpe:<15}")
            else:
                print(f"{window:<8} ❌ 读取失败")
        else:
            print(f"{window:<8} ⏳ 未开始")

    # 显示进程状态
    print(f"\n{'='*70}")
    print("【进程状态】")
    import subprocess
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        python_processes = [
            line for line in result.stdout.split('\n')
            if 'python' in line and ('run_5_windows' in line or 'main_simple' in line or 'run_baseline' in line)
        ]

        if python_processes:
            print(f"运行中的 Python 进程数: {len(python_processes)}")
        else:
            print("⚠️  没有检测到运行中的 Python 进程")

    except Exception as e:
        print(f"无法获取进程信息: {e}")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    monitor_experiment()
