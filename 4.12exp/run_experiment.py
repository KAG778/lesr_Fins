#!/usr/bin/env python3
"""
4.12exp: LESR Experiment with Layer A+B Improvements

- Layer A: Random reward detection (defensive)
- Layer B: Structured feature library for diversity

Based on exp4.7/run_amzn_nflx.py with fixes:
- Added data_pkl_path to config (fixes multi-GPU training)
- Adapted paths for 4.12exp directory
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Setup paths - project root is lesr/, FINSABER is under lesr/FINSABER
PROJECT_ROOT = Path(__file__).parent.parent  # lesr/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'FINSABER'))

# Add 4.12exp itself to path so lesr_controller etc. can be imported
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import yaml
import numpy as np

# Configure logging
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / 'run_experiment.log'))
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(pickle_file: str):
    """Load financial data."""
    from backtest.data_util.finmem_dataset import FinMemDataset
    return FinMemDataset(pickle_file=pickle_file)


def setup_environment(config):
    """Setup API keys and environment."""
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
    """Run LESR optimization with Layer A+B improvements."""
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
        'output_dir': config['output'].get('output_dir', '4.12exp/results'),
        # FIX: pass data_pkl_path for multi-GPU worker processes
        'data_pkl_path': config['data']['pickle_file'],
    }

    logger.info("Starting LESR optimization (Layer A+B)...")
    controller = LESRController(lesr_config)
    best_config = controller.run_optimization()

    return best_config


def import_module_from_file(module_name: str, file_path: Path):
    """Dynamically import a module from a file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_test_set_evaluation(config, results_dir: str, ticker: str):
    """Run test set evaluation for a specific ticker."""
    from dqn_trainer import DQNTrainer
    from baseline import train_baseline_dqn

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Test Set Evaluation for {ticker}")
    logger.info(f"{'=' * 60}")

    # Find best result for this ticker
    best_result = None
    best_iteration = None
    best_sample_id = None

    for iteration in range(3):
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

    logger.info(f"Best validation sample: iteration={best_iteration}, "
                f"sample={best_sample_id}, val_sharpe={best_result['sharpe']:.3f}")

    # Load data
    data_loader = load_data(config['data']['pickle_file'])
    train_period = tuple(config['experiment']['train_period'])
    val_period = tuple(config['experiment']['val_period'])
    test_period = tuple(config['experiment']['test_period'])

    # Load the code module
    code_file = Path(results_dir) / f'it{best_iteration}_sample{best_sample_id}.py'
    if not code_file.exists():
        logger.error(f"Code file not found: {code_file}")
        return None

    module = import_module_from_file(f'it{best_iteration}_sample{best_sample_id}', code_file)

    # Get state_dim
    test_state = np.zeros(120)
    enhanced = module.revise_state(test_state)
    state_dim = enhanced.shape[0]

    logger.info(f"State dim: {state_dim}")

    # Re-train LESR trainer
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

    # Train and evaluate baseline
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

    # Compile results
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

    # Print results
    print('\n' + '=' * 60)
    print(f'{ticker} TEST SET RESULTS')
    print('=' * 60)
    print(f'Best validation sample: iteration={best_iteration}, sample={best_sample_id}')
    print(f'Validation Sharpe: {best_result["sharpe"]:.3f}')
    print()
    print(f'LESR on {test_period[0]} to {test_period[1]}:')
    print(f"  Sharpe: {float(lesr_test['sharpe']):.3f}")
    print(f"  MaxDD: {float(lesr_test['max_dd']):.2f}%")
    print(f"  Return: {float(lesr_test['total_return']):.2f}%")
    print()
    print('Baseline on test:')
    print(f"  Sharpe: {float(baseline_test['sharpe']):.3f}")
    print(f"  MaxDD: {float(baseline_test['max_dd']):.2f}%")
    print(f"  Return: {float(baseline_test['total_return']):.2f}%")

    improvement = None
    if float(baseline_test['sharpe']) != 0:
        improvement = (float(lesr_test['sharpe']) / float(baseline_test['sharpe']) - 1) * 100
        print(f'Sharpe improvement vs baseline: {improvement:+.1f}%')

    return results


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("4.12exp: LESR with Layer A+B Improvements")
    logger.info("  Layer A: Random reward detection")
    logger.info("  Layer B: Structured feature library for diversity")
    logger.info("=" * 60)

    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    config = load_config(str(config_path))

    # Setup API key
    openai_key = setup_environment(config)

    # Load data
    logger.info("Loading financial data...")
    data_path = config['data']['pickle_file']
    # Resolve relative to project root
    if not os.path.isabs(data_path):
        data_path = str(PROJECT_ROOT / data_path)
    data_loader = load_data(data_path)
    logger.info(f"Loaded {len(data_loader.get_date_range())} trading days")
    logger.info(f"Available tickers: {data_loader.get_tickers_list()}")

    results_dir = config['output'].get('output_dir', '4.12exp/results')
    if not os.path.isabs(results_dir):
        results_dir = str(PROJECT_ROOT / results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Update config with resolved paths
    config['data']['pickle_file'] = data_path
    config['output']['output_dir'] = results_dir

    # Check if optimization has already been run
    optimization_needed = not os.path.exists(os.path.join(results_dir, 'iteration_2', 'results.pkl'))

    if optimization_needed:
        logger.info("\n" + "=" * 50)
        logger.info("Running LESR Optimization")
        logger.info("=" * 50)

        best_config = run_lesr_optimization(config, data_loader, openai_key)

        if best_config:
            logger.info(f"\nOptimization completed!")
            logger.info(f"Best strategy: Iteration {best_config['iteration']}, "
                       f"Sharpe = {best_config['sharpe']:.3f}")
        else:
            logger.warning("No valid strategy found")
            return
    else:
        logger.info("\nOptimization already completed. Running test set evaluation...")

    # Run test set evaluation for each ticker
    logger.info("\n" + "=" * 50)
    logger.info("Running Test Set Evaluation")
    logger.info("=" * 50)

    all_results = {}

    for ticker in config['experiment']['tickers']:
        ticker_results = run_test_set_evaluation(config, results_dir, ticker)
        if ticker_results:
            all_results[ticker] = ticker_results

    # Save all results
    results_file = os.path.join(results_dir, 'test_set_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"\nResults saved to {results_file}")

    # Print summary
    print('\n' + '=' * 60)
    print('FINAL SUMMARY (4.12exp - Layer A+B)')
    print('=' * 60)

    for ticker, results in all_results.items():
        print(f"\n{ticker}:")
        print(f"  LESR Sharpe: {float(results['lesr_test']['sharpe']):.3f}")
        print(f"  Baseline Sharpe: {float(results['baseline_test']['sharpe']):.3f}")
        if float(results['baseline_test']['sharpe']) != 0:
            improvement = (float(results['lesr_test']['sharpe']) /
                          float(results['baseline_test']['sharpe']) - 1) * 100
            print(f"  Improvement: {improvement:+.1f}%")


if __name__ == '__main__':
    main()
