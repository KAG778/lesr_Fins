#!/usr/bin/env python3
"""
Exp4.7 简化版主程序 - 不依赖复杂的FINSABER回测框架
专注于LESR优化流程验证
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
import numpy as np

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'FINSABER'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(pickle_file: str):
    """加载数据"""
    import sys
    # 添加FINSABER目录到Python路径
    finsaber_path = '/home/wangmeiyi/AuctionNet/lesr/FINSABER'
    if finsaber_path not in sys.path:
        sys.path.insert(0, finsaber_path)
    from backtest.data_util.finmem_dataset import FinMemDataset
    return FinMemDataset(pickle_file=pickle_file)


def setup_environment(config):
    """设置环境变量"""
    # 从config读取密钥
    if 'llm' in config and 'api_key' in config['llm']:
        api_key = config['llm']['api_key']
        if api_key and not api_key.startswith('sk-your'):
            logger.info("✓ 使用config.yaml中的API密钥")
            if 'base_url' in config['llm']:
                os.environ['OPENAI_BASE_URL'] = config['llm']['base_url']
                logger.info(f"✓ Base URL: {config['llm']['base_url']}")
            return api_key
    raise ValueError("请在config.yaml中设置llm.api_key")


def run_lesr_optimization(config, data_loader, openai_key):
    """运行LESR优化"""
    from lesr_controller import LESRController

    lesl_config = {
        'tickers': config['experiment']['tickers'],
        'train_period': config['experiment']['train_period'],
        'val_period': config['experiment']['val_period'],
        'test_period': config['experiment']['test_period'],
        'data_loader': data_loader,
        'sample_count': config['experiment'].get('sample_count', 6),
        'max_iterations': config['experiment'].get('max_iterations', 3),
        'openai_key': openai_key,
        'model': config['llm'].get('model', 'gpt-4o-mini'),
        'temperature': config['llm'].get('temperature', 0.7),
        'base_url': config['llm'].get('base_url'),
        'output_dir': config['output'].get('output_dir', 'exp4.7/results')
    }

    logger.info("开始LESR优化...")
    controller = LESRController(lesl_config)
    best_config = controller.run_optimization()

    return best_config


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Exp4.7: 金融交易LESR优化实验")
    logger.info("=" * 60)

    # 加载配置
    config = load_config('config.yaml')

    # 设置API密钥
    openai_key = setup_environment(config)

    # 加载数据
    logger.info("加载数据...")
    data_loader = load_data(config['data']['pickle_file'])
    logger.info(f"数据加载完成: {len(data_loader.get_date_range())} 个交易日")

    # 创建输出目录
    output_dir = config['output'].get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 运行优化
    try:
        best_config = run_lesr_optimization(config, data_loader, openai_key)

        if best_config:
            logger.info("=" * 50)
            logger.info("优化完成!")
            logger.info(f"最佳策略: Iteration {best_config['iteration']}, "
                       f"Sharpe = {best_config['sharpe']:.3f}")
        else:
            logger.warning("未找到有效策略")

    except Exception as e:
        logger.error(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
