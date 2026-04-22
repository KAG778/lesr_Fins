#!/usr/bin/env python3
"""
并行测试集评估: 4 stocks, 2012-2017 window
每只股票用1张A100，并行训练+评估 LESR 和 Baseline，输出结果文档
"""
import os
import sys
import pickle
import logging
from pathlib import Path
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))

SCRIPT_DIR = Path(__file__).parent

import pandas as pd
import yaml
import numpy as np
import torch
from torch.multiprocessing import Pool

os.makedirs(SCRIPT_DIR / 'logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(SCRIPT_DIR / 'logs' / 'test_eval_4stocks.log'))
    ]
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "exp4.7/result_4.8_ 训练 2012-2014 | 验证 2015-2016 | 测试 2017"


def _eval_ticker_worker(args):
    """Worker: train+eval LESR and Baseline for one ticker on one GPU."""
    ticker, gpu_id, data_pkl_path, train_start, train_end, val_start, val_end, test_start, test_end, max_episodes = args

    try:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)

        # Find best sample for this ticker across iterations
        best_result = None
        best_iteration = None
        best_sample_id = None

        for iteration in range(3):
            result_file = os.path.join(RESULTS_DIR, f'iteration_{iteration}', 'results.pkl')
            if os.path.exists(result_file):
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)
                for result in data['results']:
                    if result['ticker'] == ticker:
                        if best_result is None or float(result['sharpe']) > float(best_result['sharpe']):
                            best_result = result
                            best_iteration = iteration
                            best_sample_id = result['sample_id']

        if best_result is None:
            return {'ticker': ticker, 'error': f'No results found for {ticker}'}

        # Load code module
        import importlib.util
        code_file = Path(RESULTS_DIR) / f'it{best_iteration}_sample{best_sample_id}.py'
        if not code_file.exists():
            return {'ticker': ticker, 'error': f'Code file not found: {code_file}'}

        spec = importlib.util.spec_from_file_location(f'it{best_iteration}_sample{best_sample_id}', str(code_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get state_dim
        test_state = np.zeros(120)
        enhanced = module.revise_state(test_state)
        state_dim = enhanced.shape[0]

        # Load data
        from backtest.data_util.finmem_dataset import FinMemDataset
        from dqn_trainer import DQNTrainer
        data_loader = FinMemDataset(pickle_file=data_pkl_path)

        # --- LESR ---
        lesr_trainer = DQNTrainer(
            ticker=ticker,
            revise_state_func=module.revise_state,
            intrinsic_reward_func=module.intrinsic_reward,
            state_dim=state_dim,
            device=device
        )
        lesr_trainer.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        lesr_test = lesr_trainer.evaluate(data_loader, test_start, test_end)

        # --- Baseline ---
        def identity_revise(s):
            return s
        def zero_intrinsic(s):
            return 0.0

        baseline_trainer = DQNTrainer(
            ticker=ticker,
            revise_state_func=identity_revise,
            intrinsic_reward_func=zero_intrinsic,
            state_dim=120,
            device=device
        )
        baseline_trainer.intrinsic_weight = 0.0
        baseline_trainer.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        baseline_val = baseline_trainer.evaluate(data_loader, val_start, val_end)
        baseline_test = baseline_trainer.evaluate(data_loader, test_start, test_end)

        return {
            'ticker': ticker,
            'best_iteration': best_iteration,
            'best_sample_id': best_sample_id,
            'validation_sharpe': float(best_result['sharpe']),
            'lesr_test': lesr_test,
            'baseline_val': baseline_val,
            'baseline_test': baseline_test,
            'state_dim': state_dim,
            'code_file': str(code_file),
            'error': None
        }

    except Exception as e:
        import traceback
        return {
            'ticker': ticker,
            'error': f'{e}\n{traceback.format_exc()}'
        }


def main():
    logger.info("=" * 60)
    logger.info("Parallel Test Set Evaluation: 4 Stocks, 2012-2017")
    logger.info("=" * 60)

    config_path = str(SCRIPT_DIR / 'config_4stocks_2012_2017.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    tickers = config['experiment']['tickers']
    train_start, train_end = config['experiment']['train_period']
    val_start, val_end = config['experiment']['val_period']
    test_start, test_end = config['experiment']['test_period']
    data_pkl_path = config['data']['pickle_file']
    max_episodes = config['dqn'].get('max_episodes', 50)

    num_gpus = torch.cuda.device_count()
    logger.info(f"GPUs: {num_gpus}, Tickers: {tickers}")

    # Build tasks: each ticker on one GPU
    tasks = []
    for i, ticker in enumerate(tickers):
        gpu_id = i % num_gpus
        tasks.append((
            ticker, gpu_id, data_pkl_path,
            train_start, train_end, val_start, val_end,
            test_start, test_end, max_episodes
        ))

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    logger.info("Starting parallel evaluation...")
    with Pool(processes=len(tickers)) as pool:
        results = pool.map(_eval_ticker_worker, tasks)

    # Process results
    all_results = {}
    for r in results:
        if r.get('error'):
            logger.error(f"[{r['ticker']}] {r['error']}")
            continue
        ticker = r['ticker']
        all_results[ticker] = r
        logger.info(f"[{ticker}] LESR Sharpe={r['lesr_test']['sharpe']:.3f}, "
                    f"Baseline Sharpe={r['baseline_test']['sharpe']:.3f}")

    # Save pickle
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, 'test_set_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"Results saved to {results_file}")

    # Generate markdown report
    report = generate_report(all_results, config)
    report_file = os.path.join(RESULTS_DIR, '实验结果报告.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")

    print(report)


def generate_report(all_results, config):
    lines = []
    lines.append("# Exp4.8 LESR 实验结果报告")
    lines.append(f"\n> 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> 并行评估: 4张 NVIDIA A100")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append("| 项目 | 配置 |")
    lines.append("|------|------|")
    lines.append("| 标的 | TSLA, NFLX, AMZN, MSFT |")
    lines.append(f"| 训练期 | {config['experiment']['train_period'][0]} ~ {config['experiment']['train_period'][1]} (3年) |")
    lines.append(f"| 验证期 | {config['experiment']['val_period'][0]} ~ {config['experiment']['val_period'][1]} (2年) |")
    lines.append(f"| 测试期 | {config['experiment']['test_period'][0]} ~ {config['experiment']['test_period'][1]} (1年) |")
    lines.append("| LLM | gpt-4o-mini |")
    lines.append("| RL | DQN (50 episodes) |")
    lines.append("| 原始状态 | 120维量价数据 (20日×OHLCV) |")
    lines.append("| 初始合格样本 | ≥3 (init_min_valid) |")
    lines.append("| 迭代数 | 3 |")
    lines.append("")

    # Summary table
    lines.append("## 测试集总览 (2017全年)")
    lines.append("")
    lines.append("| 股票 | LESR Sharpe | Base Sharpe | Sharpe提升 | LESR Return | Base Return | LESR MaxDD | Base MaxDD |")
    lines.append("|------|------------|-------------|-----------|-------------|-------------|------------|------------|")

    for ticker in ['TSLA', 'NFLX', 'AMZN', 'MSFT']:
        if ticker not in all_results:
            continue
        r = all_results[ticker]
        lt = r['lesr_test']
        bt = r['baseline_test']
        ls = float(lt['sharpe'])
        bs = float(bt['sharpe'])
        imp = f"{(ls/bs - 1)*100:+.1f}%" if bs != 0 else "N/A"
        lines.append(f"| {ticker} | **{ls:.3f}** | {bs:.3f} | {imp} "
                    f"| **{lt['total_return']:.2f}%** | {bt['total_return']:.2f}% "
                    f"| {lt['max_dd']:.2f}% | {bt['max_dd']:.2f}% |")

    lines.append("")

    # Per-stock details
    lines.append("## 各股票详细结果")
    lines.append("")

    for ticker in ['TSLA', 'NFLX', 'AMZN', 'MSFT']:
        if ticker not in all_results:
            lines.append(f"### {ticker}\n\n无结果\n")
            continue
        r = all_results[ticker]
        lines.append(f"### {ticker}")
        lines.append("")
        lines.append(f"- **最佳策略**: Iteration {r['best_iteration']}, Sample {r['best_sample_id']}")
        lines.append(f"- **验证集 Sharpe**: {r['validation_sharpe']:.3f}")
        lines.append(f"- **特征维度**: 120 → {r['state_dim']}")
        lines.append("")
        lines.append("| 阶段 | Sharpe | MaxDD | Return |")
        lines.append("|------|--------|-------|--------|")
        lines.append(f"| 测试期 LESR | **{r['lesr_test']['sharpe']:.3f}** | {r['lesr_test']['max_dd']:.2f}% | **{r['lesr_test']['total_return']:.2f}%** |")
        lines.append(f"| 测试期 Baseline | {r['baseline_test']['sharpe']:.3f} | {r['baseline_test']['max_dd']:.2f}% | {r['baseline_test']['total_return']:.2f}% |")
        bs = float(r['baseline_test']['sharpe'])
        ls = float(r['lesr_test']['sharpe'])
        if bs != 0:
            lines.append(f"\n**Sharpe 提升: {(ls/bs - 1)*100:+.1f}%**")
        lines.append("")

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
