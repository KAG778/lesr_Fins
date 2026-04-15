#!/usr/bin/env python3
"""
基线训练脚本 - 与LESR并行运行

使用原始OHLCV数据(无特征工程)训练DQN作为基线对比
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'FINSABER'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exp4.7/logs/baseline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def identity_revise_state(raw_state):
    """基线：不做任何特征工程"""
    return raw_state


def zero_intrinsic_reward(state):
    """基线：无内在奖励"""
    return 0.0


def train_baseline():
    """训练基线DQN"""
    from backtest.data_util.finmem_dataset import FinMemDataset
    from dqn_trainer import DQNTrainer

    # 加载数据
    logger.info("加载数据...")
    data_loader = FinMemDataset(pickle_file="exp4.7/data/stock_data_exp4_7.pkl")

    # 配置
    tickers = ['TSLA', 'MSFT']
    train_period = ('2018-01-01', '2020-12-31')
    val_period = ('2021-01-01', '2022-12-31')

    results = {}

    for ticker in tickers:
        logger.info(f"\n{'='*50}")
        logger.info(f"训练基线: {ticker}")
        logger.info(f"{'='*50}")

        try:
            trainer = DQNTrainer(
                ticker=ticker,
                revise_state_func=identity_revise_state,
                intrinsic_reward_func=zero_intrinsic_reward,
                state_dim=120,  # 原始OHLCV
                intrinsic_weight=0.0  # 无内在奖励
            )

            # 训练
            logger.info("开始训练...")
            trainer.train(
                data_loader,
                train_period[0],
                train_period[1],
                max_episodes=50
            )

            # 验证
            logger.info("开始验证...")
            val_metrics = trainer.evaluate(
                data_loader,
                val_period[0],
                val_period[1]
            )

            results[ticker] = {
                'sharpe': val_metrics['sharpe'],
                'max_dd': val_metrics['max_dd'],
                'total_return': val_metrics['total_return'],
                'trainer': trainer
            }

            logger.info(f"{ticker} 基线结果:")
            logger.info(f"  Sharpe: {val_metrics['sharpe']:.3f}")
            logger.info(f"  Max DD: {val_metrics['max_dd']:.2f}%")
            logger.info(f"  Total Return: {val_metrics['total_return']:.2f}%")

        except Exception as e:
            logger.error(f"{ticker} 基线训练失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    os.makedirs('exp4.7/results', exist_ok=True)
    with open('exp4.7/results/baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    logger.info("\n" + "="*50)
    logger.info("基线训练完成!")
    logger.info("="*50)

    return results


if __name__ == '__main__':
    train_baseline()
