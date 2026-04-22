"""
Main Entry Point for Exp4.9: Regime-Conditioned LESR

Changes from 4.7:
- Uses regime-aware controller and strategy
- Reports per-regime metrics in test results
"""

import os
import sys
import argparse
import pickle
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.data_util.finmem_dataset import FinMemDataset
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
sys.path.insert(0, str(Path(__file__).parent))

from lesr_controller import LESRController
from lesr_strategy import LESRStrategy
from baseline import BaselineDQNStrategy, train_baseline_dqn

# Configure logging
os.makedirs('exp4.9/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exp4.9/logs/exp4.9.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_environment(config):
    openai_key = None
    base_url = None

    if 'llm' in config and 'api_key' in config['llm']:
        key = config['llm']['api_key']
        if key and not key.startswith('sk-your'):
            openai_key = key
            logger.info("Using API key from config.yaml")

    if not openai_key:
        key_file = Path('exp4.9/.api_keys.txt')
        if key_file.exists():
            with open(key_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENAI_API_KEY='):
                        openai_key = line.split('=', 1)[1].strip()
                    elif line.startswith('OPENAI_BASE_URL='):
                        base_url = line.split('=', 1)[1].strip()

    if not openai_key:
        openai_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL', base_url)

    if not openai_key:
        raise ValueError("No API key found!")

    if 'llm' in config and 'base_url' in config['llm']:
        base_url = config['llm']['base_url'] or base_url

    if base_url:
        os.environ['OPENAI_BASE_URL'] = base_url
        logger.info(f"Base URL: {base_url}")

    return openai_key


def main():
    parser = argparse.ArgumentParser(description='Exp4.9: Regime-Conditioned LESR')
    parser.add_argument('--config', type=str, default='exp4.9/config.yaml')
    parser.add_argument('--skip_optimization', action='store_true')
    parser.add_argument('--results_dir', type=str)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Exp4.9: Regime-Conditioned LESR for Financial Trading")
    logger.info("=" * 60)

    config = load_config(args.config)
    openai_key = setup_environment(config)

    output_dir = config.get('output', {}).get('output_dir', 'exp4.9/results')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info("Loading financial data...")
    data_loader = FinMemDataset(pickle_file=config['data']['pickle_file'])
    logger.info(f"Loaded {len(data_loader.get_date_range())} trading days")

    # Configure controller
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
        'base_url': config['llm'].get('base_url') or os.getenv('OPENAI_BASE_URL'),
        'output_dir': output_dir,
        'data_pkl_path': config['data']['pickle_file']
    }

    # Run LESR optimization
    if args.skip_optimization and args.results_dir:
        logger.info(f"Loading existing results from {args.results_dir}")
        best_config = load_best_config(args.results_dir)
    else:
        logger.info("Running Regime-Conditioned LESR optimization...")
        controller = LESRController(lesr_config)
        best_config = controller.run_optimization()

    if best_config is None:
        logger.error("No valid strategy found.")
        return

    # Test set backtesting
    logger.info("\n" + "=" * 50)
    logger.info("Testing on Test Set")
    logger.info("=" * 50)

    # LESR Strategy
    logger.info("\n--- LESR Strategy (Regime-Conditioned) ---")
    lesr_strategy = LESRStrategy(
        ticker=best_config['ticker'],
        revise_state_func=best_config['trainer'].revise_state,
        trained_dqn=best_config['trainer'].dqn,
        enable_safety_net=True
    )

    framework = FINSABERFrameworkHelper(
        initial_cash=config['backtest'].get('initial_cash', 100000),
        commission_per_share=config['backtest'].get('commission', 0.001),
        min_commission=config['backtest'].get('min_commission', 0.99)
    )

    framework.load_backtest_data_single_ticker(
        data_loader, best_config['ticker'],
        start_date=pd.to_datetime(lesr_config['test_period'][0]),
        end_date=pd.to_datetime(lesr_config['test_period'][1])
    )

    framework.run(lesr_strategy)
    lesr_metrics = framework.evaluate(lesr_strategy)

    logger.info(f"\nLESR Results:")
    logger.info(f"  Sharpe: {lesr_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Max DD: {lesr_metrics['max_drawdown']:.2f}%")
    logger.info(f"  Total Return: {lesr_metrics['total_return']*100:.2f}%")

    # Baseline (UNCHANGED from 4.7)
    logger.info("\n--- Baseline Strategy ---")
    baseline_trainer, baseline_val = train_baseline_dqn(
        ticker=best_config['ticker'],
        data_loader=data_loader,
        train_period=lesr_config['train_period'],
        val_period=lesr_config['val_period']
    )

    baseline_strategy = BaselineDQNStrategy(
        ticker=best_config['ticker'],
        trained_dqn=baseline_trainer.dqn
    )

    framework.reset()
    framework.load_backtest_data_single_ticker(
        data_loader, best_config['ticker'],
        start_date=pd.to_datetime(lesr_config['test_period'][0]),
        end_date=pd.to_datetime(lesr_config['test_period'][1])
    )

    framework.run(baseline_strategy)
    baseline_metrics = framework.evaluate(baseline_strategy)

    logger.info(f"\nBaseline Results:")
    logger.info(f"  Sharpe: {baseline_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Max DD: {baseline_metrics['max_drawdown']:.2f}%")
    logger.info(f"  Total Return: {baseline_metrics['total_return']*100:.2f}%")

    # Comparison
    logger.info("\n" + "=" * 50)
    logger.info("Comparison: LESR (Regime-Conditioned) vs Baseline")
    logger.info("=" * 50)

    sharpe_imp = (
        (lesr_metrics['sharpe_ratio'] / baseline_metrics['sharpe_ratio'] - 1) * 100
        if baseline_metrics['sharpe_ratio'] > 0 else 0
    )

    logger.info(f"LESR Sharpe:     {lesr_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Baseline Sharpe: {baseline_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Improvement:     {sharpe_imp:+.1f}%")
    logger.info(f"\nLESR Max DD:     {lesr_metrics['max_drawdown']:.2f}%")
    logger.info(f"Baseline Max DD: {baseline_metrics['max_drawdown']:.2f}%")
    logger.info(f"\nLESR Return:     {lesr_metrics['total_return']*100:.2f}%")
    logger.info(f"Baseline Return: {baseline_metrics['total_return']*100:.2f}%")

    # Save results
    results = {
        'lesr': lesr_metrics,
        'baseline': baseline_metrics,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'experiment': 'exp4.9_regime_conditioned'
    }

    results_file = os.path.join(output_dir, 'final_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    # Save summary
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Exp4.9: Regime-Conditioned LESR - Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Ticker: {best_config['ticker']}\n\n")
        f.write("LESR Strategy (Regime-Conditioned):\n")
        f.write(f"  Sharpe: {lesr_metrics['sharpe_ratio']:.3f}\n")
        f.write(f"  Max DD: {lesr_metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Total Return: {lesr_metrics['total_return']*100:.2f}%\n\n")
        f.write("Baseline Strategy:\n")
        f.write(f"  Sharpe: {baseline_metrics['sharpe_ratio']:.3f}\n")
        f.write(f"  Max DD: {baseline_metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Total Return: {baseline_metrics['total_return']*100:.2f}%\n\n")
        f.write(f"Sharpe Improvement: {sharpe_imp:+.1f}%\n")

    logger.info(f"\nResults saved to {results_file}")


def load_best_config(results_dir: str) -> dict:
    best_sharpe = -float('inf')
    best_config = None

    for iteration_dir in Path(results_dir).glob('iteration_*'):
        result_file = iteration_dir / 'results.pkl'
        if result_file.exists():
            with open(result_file, 'rb') as f:
                data = pickle.load(f)
            for result in data['results']:
                if result['sharpe'] > best_sharpe:
                    best_sharpe = result['sharpe']
                    best_config = result

    return best_config


if __name__ == '__main__':
    main()
