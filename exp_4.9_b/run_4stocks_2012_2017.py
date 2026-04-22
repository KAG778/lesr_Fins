#!/usr/bin/env python3
"""
Run LESR Pipeline for 4 stocks on 2012-2017 time window
训练 2012-2014 | 验证 2015-2016 | 测试 2017
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))

SCRIPT_DIR = Path(__file__).parent

import pandas as pd
import yaml
import numpy as np

os.makedirs(SCRIPT_DIR / 'logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(SCRIPT_DIR / 'logs' / 'run_4stocks_2012_2017.log'))
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(pickle_file):
    from backtest.data_util.finmem_dataset import FinMemDataset
    return FinMemDataset(pickle_file=pickle_file)


def setup_environment(config):
    if 'llm' in config and 'api_key' in config['llm']:
        api_key = config['llm']['api_key']
        if api_key and not api_key.startswith('sk-your'):
            logger.info("Using API key from config")
            if 'base_url' in config['llm']:
                os.environ['OPENAI_BASE_URL'] = config['llm']['base_url']
                logger.info(f"Base URL: {config['llm']['base_url']}")
            return api_key
    raise ValueError("Please set llm.api_key in config file")


def run_lesr_optimization(config, data_loader, openai_key):
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
        'openai_key': openai_key,
        'model': config['llm'].get('model', 'gpt-4o-mini'),
        'temperature': config['llm'].get('temperature', 0.7),
        'base_url': config['llm'].get('base_url'),
        'output_dir': config['output'].get('output_dir', 'exp4.7/result_4.8'),
        'data_pkl_path': config['data']['pickle_file'],
    }

    logger.info("Starting LESR optimization...")
    controller = LESRController(lesr_config)
    best_config = controller.run_optimization()
    return best_config


def import_module_from_file(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_test_set_evaluation(config, results_dir, ticker):
    from dqn_trainer import DQNTrainer
    from baseline import train_baseline_dqn

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Test Set Evaluation for {ticker}")
    logger.info(f"{'=' * 60}")

    best_result = None
    best_iteration = None
    best_sample_id = None
    max_iter = config['experiment'].get('max_iterations', 3)

    for iteration in range(max_iter):
        result_file = os.path.join(results_dir, f'iteration_{iteration}', 'results.pkl')
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
        logger.warning(f"No results found for {ticker}")
        return None

    logger.info(f"Best: iteration={best_iteration}, sample={best_sample_id}, val_sharpe={best_result['sharpe']:.3f}")

    # Load code module
    code_file = Path(results_dir) / f'it{best_iteration}_sample{best_sample_id}.py'
    if not code_file.exists():
        logger.error(f"Code file not found: {code_file}")
        return None

    module = import_module_from_file(f'it{best_iteration}_sample{best_sample_id}', code_file)

    # Get state_dim
    iter_file = Path(results_dir) / f'iteration_{best_iteration}' / 'results.pkl'
    with open(iter_file, 'rb') as f:
        iter_data = pickle.load(f)

    state_dim = None
    for sample in iter_data['samples']:
        if iter_data['samples'].index(sample) == best_sample_id:
            state_dim = sample['state_dim']
            break
    if state_dim is None:
        test_state = np.zeros(120)
        enhanced = module.revise_state(test_state)
        state_dim = enhanced.shape[0]

    logger.info(f"State dim: {state_dim}")

    train_period = tuple(config['experiment']['train_period'])
    val_period = tuple(config['experiment']['val_period'])
    test_period = tuple(config['experiment']['test_period'])

    # Load data
    from backtest.data_util.finmem_dataset import FinMemDataset
    data_loader = FinMemDataset(pickle_file=config['data']['pickle_file'])

    # Re-train LESR
    logger.info(f"Training LESR model for {ticker}...")
    lesr_trainer = DQNTrainer(
        ticker=ticker,
        revise_state_func=module.revise_state,
        intrinsic_reward_func=module.intrinsic_reward,
        state_dim=state_dim,
    )
    lesr_trainer.train(data_loader, train_period[0], train_period[1],
                       max_episodes=config['dqn'].get('max_episodes', 50))
    logger.info(f"Evaluating LESR on test period...")
    lesr_test = lesr_trainer.evaluate(data_loader, test_period[0], test_period[1])

    # Baseline
    logger.info(f"Training baseline model for {ticker}...")
    baseline_trainer, baseline_val = train_baseline_dqn(
        ticker=ticker,
        data_loader=data_loader,
        train_period=train_period,
        val_period=val_period,
        state_dim=120,
        intrinsic_weight=0.0,
    )
    logger.info(f"Evaluating baseline on test period...")
    baseline_test = baseline_trainer.evaluate(data_loader, test_period[0], test_period[1])

    results = {
        'ticker': ticker,
        'best_iteration': best_iteration,
        'best_sample_id': best_sample_id,
        'validation_sharpe': best_result['sharpe'],
        'test_period': test_period,
        'lesr_test': lesr_test,
        'baseline_val': baseline_val,
        'baseline_test': baseline_test,
        'code_file': str(code_file),
    }

    print(f'\n{ticker} TEST SET RESULTS')
    print(f'  Val Sharpe: {best_result["sharpe"]:.3f}')
    print(f'  LESR  - Sharpe: {lesr_test["sharpe"]:.3f}, MaxDD: {lesr_test["max_dd"]:.2f}%, Return: {lesr_test["total_return"]:.2f}%')
    print(f'  Base  - Sharpe: {baseline_test["sharpe"]:.3f}, MaxDD: {baseline_test["max_dd"]:.2f}%, Return: {baseline_test["total_return"]:.2f}%')
    if float(baseline_test['sharpe']) != 0:
        imp = (float(lesr_test['sharpe']) / float(baseline_test['sharpe']) - 1) * 100
        print(f'  Sharpe improvement: {imp:+.1f}%')

    return results


def main():
    logger.info("=" * 60)
    logger.info("Exp4.8: 4 Stocks, 2012-2017 Window")
    logger.info("Train 2012-2014 | Val 2015-2016 | Test 2017")
    logger.info("=" * 60)

    os.makedirs(SCRIPT_DIR / 'logs', exist_ok=True)

    config = load_config(str(SCRIPT_DIR / 'config_4stocks_2012_2017.yaml'))
    openai_key = setup_environment(config)

    logger.info("Loading financial data...")
    data_loader = load_data(config['data']['pickle_file'])
    logger.info(f"Available tickers: {data_loader.get_tickers_list()}")
    logger.info(f"Date range: {len(data_loader.get_date_range())} trading days")

    results_dir = config['output'].get('output_dir')
    os.makedirs(results_dir, exist_ok=True)

    # Check if optimization already done
    max_iter = config['experiment'].get('max_iterations', 3)
    optimization_needed = not os.path.exists(
        os.path.join(results_dir, f'iteration_{max_iter - 1}', 'results.pkl')
    )

    if optimization_needed:
        logger.info("\nRunning LESR Optimization...")
        best_config = run_lesr_optimization(config, data_loader, openai_key)
        if best_config:
            logger.info(f"Optimization completed! Best: It {best_config['iteration']}, Sharpe={best_config['sharpe']:.3f}")
        else:
            logger.warning("No valid strategy found")
            return
    else:
        logger.info("Optimization already completed. Running test evaluation...")

    # Test set evaluation
    logger.info("\nRunning Test Set Evaluation...")
    all_results = {}
    for ticker in config['experiment']['tickers']:
        ticker_results = run_test_set_evaluation(config, results_dir, ticker)
        if ticker_results:
            all_results[ticker] = ticker_results

    # Save
    results_file = os.path.join(results_dir, 'test_set_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"Results saved to {results_file}")

    # Summary
    print('\n' + '=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    for ticker, res in all_results.items():
        lesr_s = float(res['lesr_test']['sharpe'])
        base_s = float(res['baseline_test']['sharpe'])
        imp = (lesr_s / base_s - 1) * 100 if base_s != 0 else float('inf')
        print(f"\n{ticker}:")
        print(f"  LESR:  Sharpe={lesr_s:.3f}, MaxDD={res['lesr_test']['max_dd']:.2f}%, Return={res['lesr_test']['total_return']:.2f}%")
        print(f"  Base:  Sharpe={base_s:.3f}, MaxDD={res['baseline_test']['max_dd']:.2f}%, Return={res['baseline_test']['total_return']:.2f}%")
        print(f"  Improvement: {imp:+.1f}%")


if __name__ == '__main__':
    main()
