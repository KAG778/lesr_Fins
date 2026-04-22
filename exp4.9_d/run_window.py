#!/usr/bin/env python3
"""
exp4.9_d Pipeline: Fixed features + LLM designs reward only
用法: python run_window.py --config exp4.9_d/config_W1.yaml
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

import yaml
import numpy as np
import torch
from torch.multiprocessing import Pool

os.makedirs(SCRIPT_DIR / 'logs', exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def load_config(path):
    with open(path) as f:
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
    ticker, gpu_id, data_pkl_path, train_start, train_end, val_start, val_end, \
        test_start, test_end, results_dir, max_episodes, intrinsic_weight, commission = args
    try:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)

        # Find best reward function
        best_result = None
        for iteration in range(3):
            rf = os.path.join(results_dir, f'iteration_{iteration}', 'results.pkl')
            if os.path.exists(rf):
                with open(rf, 'rb') as f:
                    data = pickle.load(f)
                for r in data['results']:
                    if r['ticker'] == ticker:
                        if best_result is None or r['sharpe'] > best_result['sharpe']:
                            best_result = r
                            best_it = iteration
                            best_sid = r['sample_id']

        if best_result is None:
            return {'ticker': ticker, 'error': 'No results'}

        # Load reward module
        import importlib.util
        code_file = Path(results_dir) / f'it{best_it}_sample{best_sid}.py'
        spec = importlib.util.spec_from_file_location(f'best_{ticker}', str(code_file))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        from backtest.data_util.finmem_dataset import FinMemDataset
        from dqn_trainer import DQNTrainer
        from feature_engine import revise_state, STATE_DIM

        data_loader = FinMemDataset(pickle_file=data_pkl_path)

        # LESR retrain + test
        lesr = DQNTrainer(ticker, revise_state, module.intrinsic_reward, STATE_DIM,
                          intrinsic_weight=intrinsic_weight, commission=commission, device=device)
        lesr.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        lesr_test = lesr.evaluate(data_loader, test_start, test_end)

        # Baseline (same features, no intrinsic reward)
        base = DQNTrainer(ticker, revise_state, lambda s: 0.0, STATE_DIM,
                          intrinsic_weight=0.0, commission=commission, device=device)
        base.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        base_test = base.evaluate(data_loader, test_start, test_end)

        return {
            'ticker': ticker, 'best_iteration': best_it,
            'best_sample_id': best_sid,
            'validation_sharpe': float(best_result['sharpe']),
            'lesr_test': lesr_test, 'baseline_test': base_test,
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
    iw = config['intrinsic'].get('weight', 0.1)
    comm = config['backtest'].get('commission', 0.001)
    num_gpus = torch.cuda.device_count()

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    tasks = [(t, i % num_gpus, data_pkl, train_s, train_e, val_s, val_e,
              test_s, test_e, results_dir, max_ep, iw, comm)
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
        logger.info(f"[{r['ticker']}] LESR S={lt['sharpe']:.3f} R={lt['total_return']:.2f}% T={lt.get('num_trades',0)} | "
                    f"Base S={bt['sharpe']:.3f} R={bt['total_return']:.2f}% T={bt.get('num_trades',0)}")

    pkl_file = os.path.join(results_dir, 'test_set_results.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(all_results, f)

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
        f"# LESR exp4.9_d 实验结果",
        f"\n> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> GPU: {torch.cuda.device_count()}\n",
        "## 设置\n",
        "| 项目 | 配置 |",
        "|------|------|",
        f"| 标的 | {', '.join(config['experiment']['tickers'])} |",
        f"| 训练 | {tp[0]} ~ {tp[1]} |",
        f"| 验证 | {vp[0]} ~ {vp[1]} |",
        f"| 测试 | {tsp[0]} ~ {tsp[1]} |",
        f"| 特征 | 固定30维预计算 |",
        f"| LLM | {config['llm'].get('model', 'gpt-4o-mini')} (只设计reward) |",
        f"| intrinsic_weight | {config['intrinsic'].get('weight', 0.1)} |",
        "\n## 测试集\n",
        "| 股票 | LESR S | Base S | LESR R | Base R | LESR DD | Base DD | LESR T | Base T | 结论 |",
        "|------|--------|--------|--------|--------|---------|---------|--------|--------|------|",
    ]
    wins = 0
    total = 0
    for t in config['experiment']['tickers']:
        if t not in all_results:
            continue
        r = all_results[t]
        lt, bt = r['lesr_test'], r['baseline_test']
        ls, bs = lt['sharpe'], bt['sharpe']
        lt_t = lt.get('num_trades', len(lt.get('trades', [])) // 2)
        bt_t = bt.get('num_trades', len(bt.get('trades', [])) // 2)
        total += 1
        if ls > bs:
            wins += 1
            v = "LESR胜 ✓"
        elif bs > ls:
            v = "Base胜"
        else:
            v = "平局"
        lines.append(f"| {t} | **{ls:.3f}** | {bs:.3f} | **{lt['total_return']:.2f}%** | "
                     f"{bt['total_return']:.2f}% | {lt['max_dd']:.2f}% | {bt['max_dd']:.2f}% | "
                     f"{lt_t} | {bt_t} | {v} |")
    lines.append(f"\n**胜率: {wins}/{total}**")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = setup_env(config)

    window_name = Path(config['output']['output_dir']).name
    log_file = str(SCRIPT_DIR / 'logs' / f'{window_name}.log')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    logger.info(f"Window: {window_name}")
    logger.info(f"Train: {config['experiment']['train_period']}")
    logger.info(f"Val: {config['experiment']['val_period']}")
    logger.info(f"Test: {config['experiment']['test_period']}")

    from backtest.data_util.finmem_dataset import FinMemDataset
    data_loader = FinMemDataset(pickle_file=config['data']['pickle_file'])

    results_dir = config['output']['output_dir']
    os.makedirs(results_dir, exist_ok=True)

    max_iter = config['experiment'].get('max_iterations', 3)
    if not os.path.exists(os.path.join(results_dir, f'iteration_{max_iter-1}', 'results.pkl')):
        logger.info("Running optimization...")
        run_lesr(config, data_loader, api_key)
    else:
        logger.info("Optimization done.")

    logger.info("Running test eval...")
    run_test_eval(config, results_dir)
    logger.info("Done!")


if __name__ == '__main__':
    main()
