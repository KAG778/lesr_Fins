"""
Main Entry Point for Portfolio Optimization with LESR

Usage:
    python main.py --config configs/config.yaml --experiment_name my_experiment

LESR 投资组合优化主入口。

用法：
    python main.py --config configs/config.yaml --experiment_name my_experiment

流程：
    1. 加载 YAML 配置文件
    2. 创建 LESRController 实例
    3. 运行完整的迭代优化循环
    4. 输出最终测试结果和基线对比
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from lesr_controller import LESRController


def main():
    parser = argparse.ArgumentParser(description='LESR Portfolio Optimization')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--experiment_name', type=str, default='lesr_portfolio',
                        help='Experiment name for results directory')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create controller and run
    experiment_dir = str(Path('results') / args.experiment_name)
    controller = LESRController(config, experiment_dir)
    controller.run()


if __name__ == '__main__':
    main()
