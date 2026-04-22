#!/usr/bin/env python3
"""
通用 LESR Pipeline for exp_4.9_b: 接受配置文件路径参数
用法: python run_window.py --config exp_4.9_b/config_W1.yaml
"""
import os
import sys
import pickle
import logging
import argparse
from pathlib import Path
import datetime

sys.path.insert(0, str(Path(__file__).parent))
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
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_env(config):
    api_key = config['llm']['api_key']
    if config['llm'].get('base_url'):
        os.environ['OPENAI_BASE_URL'] = config['llm']['base_url']
    return api_key


def run_lesr(config, data_loader, api_key):
    from lesr_controller import LESRController

    lesr_config = {
        'tickers': config['experiment']['tickers'],
        'train_period': config['experiment']['train_period'],
        'val_period': config['experiment']['val_period'],
        'test_period': config['experiment']['test_period'],
        'data_loader': data_loader,
        'sample_count': config['experiment'].get('sample_count', 6),
        'max_iterations': config['experiment'].get('max_iterations', 3),
        'init_min_valid': config['experiment'].get('init_min_valid', 3),
        'init_max_rounds': config['experiment'].get('init_max_rounds', 5),
        'openai_key': api_key,
        'model': config['llm'].get('model', 'gpt-4o-mini'),
        'temperature': config['llm'].get('temperature', 0.7),
        'base_url': config['llm'].get('base_url'),
        'output_dir': config['output']['output_dir'],
        'data_pkl_path': config['data']['pickle_file'],
        'intrinsic_weight': config['intrinsic'].get('weight', 0.1),
        'commission': config['backtest'].get('commission', 0.001),
    }
    controller = LESRController(lesr_config)
    return controller.run_optimization()


def _eval_worker(args):
    ticker, gpu_id, data_pkl_path, train_start, train_end, val_start, val_end, test_start, test_end, results_dir, max_episodes, intrinsic_weight, commission = args
    try:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)

        # Find best
        best_result = None
        best_iteration = 0
        best_sample_id = 0
        for iteration in range(3):
            rf = os.path.join(results_dir, f'iteration_{iteration}', 'results.pkl')
            if os.path.exists(rf):
                with open(rf, 'rb') as f:
                    data = pickle.load(f)
                for r in data['results']:
                    if r['ticker'] == ticker:
                        if best_result is None or float(r['sharpe']) > float(best_result['sharpe']):
                            best_result = r
                            best_iteration = iteration
                            best_sample_id = r['sample_id']

        if best_result is None:
            return {'ticker': ticker, 'error': 'No results'}

        # Load module
        import importlib.util
        code_file = Path(results_dir) / f'it{best_iteration}_sample{best_sample_id}.py'
        spec = importlib.util.spec_from_file_location(f'best_{ticker}', str(code_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        test_state = np.zeros(120)
        enhanced = module.revise_state(test_state)
        state_dim = enhanced.shape[0]  # without position flag

        from backtest.data_util.finmem_dataset import FinMemDataset
        from dqn_trainer import DQNTrainer
        data_loader = FinMemDataset(pickle_file=data_pkl_path)

        # LESR (DQNTrainer adds +1 for position flag internally)
        lesr = DQNTrainer(ticker, module.revise_state, module.intrinsic_reward, state_dim,
                          intrinsic_weight=intrinsic_weight, commission=commission, device=device)
        lesr.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        lesr_test = lesr.evaluate(data_loader, test_start, test_end)

        # Baseline (120 raw dims, no features, DQNTrainer adds +1 for position flag)
        base = DQNTrainer(ticker, lambda s: s, lambda s: 0.0, 120,
                          intrinsic_weight=0.0, commission=commission, device=device)
        base.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        base_val = base.evaluate(data_loader, val_start, val_end)
        base_test = base.evaluate(data_loader, test_start, test_end)

        return {
            'ticker': ticker, 'best_iteration': best_iteration,
            'best_sample_id': best_sample_id,
            'validation_sharpe': float(best_result['sharpe']),
            'lesr_test': lesr_test, 'baseline_val': base_val,
            'baseline_test': base_test, 'state_dim': state_dim,
            'error': None
        }
    except Exception as e:
        import traceback
        return {'ticker': ticker, 'error': f'{e}\n{traceback.format_exc()}'}


def run_test_eval(config, results_dir):
    tickers = config['experiment']['tickers']
    data_pkl = config['data']['pickle_file']
    train_s, train_e = config['experiment']['train_period']
    val_s, val_e = config['experiment']['val_period']
    test_s, test_e = config['experiment']['test_period']
    max_ep = config['dqn'].get('max_episodes', 50)
    intrinsic_weight = config['intrinsic'].get('weight', 0.1)
    commission = config['backtest'].get('commission', 0.001)
    num_gpus = torch.cuda.device_count()

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    tasks = [(t, i % num_gpus, data_pkl, train_s, train_e, val_s, val_e,
              test_s, test_e, results_dir, max_ep, intrinsic_weight, commission)
             for i, t in enumerate(tickers)]

    with Pool(processes=len(tickers)) as pool:
        results = pool.map(_eval_worker, tasks)

    all_results = {}
    for r in results:
        if r.get('error'):
            logger.error(f"[{r['ticker']}] {r['error']}")
            continue
        all_results[r['ticker']] = r
        lt = r['lesr_test']
        bt = r['baseline_test']
        logger.info(f"[{r['ticker']}] LESR Sharpe={lt['sharpe']:.3f} Return={lt['total_return']:.2f}% Trades={lt.get('num_trades',0)} | "
                    f"Base Sharpe={bt['sharpe']:.3f} Return={bt['total_return']:.2f}% Trades={bt.get('num_trades',0)}")

    # Save
    pkl_file = os.path.join(results_dir, 'test_set_results.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(all_results, f)

    # Report
    report = gen_report(all_results, config)
    md_file = os.path.join(results_dir, '实验结果报告.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved: {md_file}")
    print(report)


def gen_report(all_results, config):
    tp = config['experiment']['train_period']
    vp = config['experiment']['val_period']
    tsp = config['experiment']['test_period']
    lines = [
        f"# LESR exp_4.9_b 实验结果",
        f"\n> 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 并行评估: {torch.cuda.device_count()}张 GPU\n",
        "## 实验设置\n",
        "| 项目 | 配置 |",
        "|------|------|",
        f"| 标的 | {', '.join(config['experiment']['tickers'])} |",
        f"| 训练期 | {tp[0]} ~ {tp[1]} |",
        f"| 验证期 | {vp[0]} ~ {vp[1]} |",
        f"| 测试期 | {tsp[0]} ~ {tsp[1]} |",
        f"| LLM | {config['llm'].get('model', 'gpt-4o-mini')} |",
        f"| intrinsic_weight | {config['intrinsic'].get('weight', 0.1)} |",
        f"| 迭代数 | {config['experiment'].get('max_iterations', 3)} |",
        "\n## 测试集结果\n",
        "| 股票 | LESR Sharpe | Base Sharpe | LESR Return | Base Return | LESR MaxDD | Base MaxDD | LESR Trades | Base Trades | 结论 |",
        "|------|------------|-------------|-------------|-------------|------------|------------|-------------|-------------|------|",
    ]
    lesr_wins = 0
    total = 0
    for t in config['experiment']['tickers']:
        if t not in all_results:
            continue
        r = all_results[t]
        lt, bt = r['lesr_test'], r['baseline_test']
        ls, bs = float(lt['sharpe']), float(bt['sharpe'])
        lt_trades = lt.get('num_trades', len(lt.get('trades', [])) // 2)
        bt_trades = bt.get('num_trades', len(bt.get('trades', [])) // 2)
        total += 1

        if ls > bs:
            verdict = "LESR胜 ✓"
            lesr_wins += 1
        elif bs > ls:
            verdict = "Base胜"
        else:
            verdict = "平局"

        lines.append(
            f"| {t} | **{ls:.3f}** | {bs:.3f} | **{lt['total_return']:.2f}%** | "
            f"{bt['total_return']:.2f}% | {lt['max_dd']:.2f}% | {bt['max_dd']:.2f}% | "
            f"{lt_trades} | {bt_trades} | {verdict} |"
        )

    lines.append(f"\n**胜率: {lesr_wins}/{total}**")
    lines.append("")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = setup_env(config)

    # Setup per-window logging
    window_name = Path(config['output']['output_dir']).name
    log_file = str(SCRIPT_DIR / 'logs' / f'{window_name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Window: {window_name}")
    logger.info(f"Train: {config['experiment']['train_period']}")
    logger.info(f"Val: {config['experiment']['val_period']}")
    logger.info(f"Test: {config['experiment']['test_period']}")
    logger.info(f"intrinsic_weight: {config['intrinsic'].get('weight', 0.1)}")

    from backtest.data_util.finmem_dataset import FinMemDataset
    data_loader = FinMemDataset(pickle_file=config['data']['pickle_file'])
    logger.info(f"Data: {len(data_loader.get_date_range())} days, tickers={data_loader.get_tickers_list()}")

    results_dir = config['output']['output_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Optimization
    max_iter = config['experiment'].get('max_iterations', 3)
    if not os.path.exists(os.path.join(results_dir, f'iteration_{max_iter-1}', 'results.pkl')):
        logger.info("Running optimization...")
        run_lesr(config, data_loader, api_key)
    else:
        logger.info("Optimization done, skipping.")

    # Test eval
    logger.info("Running test evaluation...")
    run_test_eval(config, results_dir)
    logger.info("Done!")


if __name__ == '__main__':
    main()
