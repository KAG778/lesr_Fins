"""
Main Entry Point for Exp4.7 Financial Trading Experiment

This script runs the complete LESR optimization pipeline:
1. Load financial data
2. Run LESR optimization (LLM + DQN)
3. Backtest best strategy on test set
4. Compare with baseline

Usage:
    python main.py --config exp4.7/config.yaml

Environment Variables:
    OPENAI_API_KEY: OpenAI API key for LLM access
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

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'FINSABER'))

from backtest.data_util.finmem_dataset import FinMemDataset
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

from lesr_controller import LESRController
from lesr_strategy import LESRStrategy
from baseline import BaselineDQNStrategy, train_baseline_dqn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exp4.7/logs/exp4.7.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config):
    """Setup environment variables and check dependencies."""
    openai_key = None
    base_url = None

    # 优先级1: 从config.yaml读取密钥
    if 'llm' in config and 'api_key' in config['llm']:
        api_key_from_config = config['llm']['api_key']
        if api_key_from_config and not api_key_from_config.startswith('sk-your'):
            openai_key = api_key_from_config
            logger.info("✓ 使用config.yaml中的API密钥")

    # 优先级2: 从.api_keys.txt文件读取
    if not openai_key:
        key_file = Path('exp4.7/.api_keys.txt')
        if key_file.exists():
            with open(key_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENAI_API_KEY='):
                        openai_key = line.split('=', 1)[1].strip()
                    elif line.startswith('OPENAI_BASE_URL='):
                        base_url = line.split('=', 1)[1].strip()
            if openai_key:
                logger.info("✓ 使用.api_keys.txt中的API密钥")

    # 优先级3: 环境变量
    if not openai_key:
        openai_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL', base_url)
        if openai_key:
            logger.info("✓ 使用环境变量中的API密钥")

    # 检查密钥
    if not openai_key:
        raise ValueError(
            "未找到API密钥！请通过以下方式之一设置:\n"
            "  1. 在config.yaml中设置llm.api_key\n"
            "  2. 运行: python exp4.7/setup_keys.py\n"
            "  3. 设置环境变量: export OPENAI_API_KEY=your_key"
        )

    # 如果config中有base_url，优先使用
    if 'llm' in config and 'base_url' in config['llm']:
        base_url = config['llm']['base_url'] or base_url

    if base_url:
        os.environ['OPENAI_BASE_URL'] = base_url
        logger.info(f"✓ Base URL: {base_url}")

    return openai_key


def main():
    parser = argparse.ArgumentParser(description='Exp4.7 Financial Trading Experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='exp4.7/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip_optimization',
        action='store_true',
        help='Skip LESR optimization and use existing results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        help='Path to existing results directory (for --skip_optimization)'
    )
    args = parser.parse_args()

    # Setup
    logger.info("=" * 60)
    logger.info("Exp4.7: Financial Trading with LESR Framework")
    logger.info("=" * 60)

    openai_key = setup_environment(config)
    config = load_config(args.config)

    # Create output directory
    output_dir = config.get('output_dir', 'exp4.7/results')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('exp4.7/logs', exist_ok=True)

    # Load data
    logger.info("Loading financial data...")
    data_loader = FinMemDataset(
        pickle_file=config['data']['pickle_file']
    )
    logger.info(f"Loaded data with {len(data_loader.get_date_range())} trading days")
    logger.info(f"Available tickers: {data_loader.get_tickers_list()[:10]}...")

    # Configure LESR controller
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
        'base_url': config['llm'].get('base_url') or os.getenv('OPENAI_BASE_URL'),
        'output_dir': output_dir
    }

    # Run LESR optimization (or load existing results)
    if args.skip_optimization and args.results_dir:
        logger.info(f"Loading existing results from {args.results_dir}")
        best_config = load_best_config(args.results_dir)
    else:
        logger.info("Running LESR optimization...")
        controller = LESRController(lesl_config)
        best_config = controller.run_optimization()

    if best_config is None:
        logger.error("No valid strategy found. Exiting.")
        return

    # Test set backtesting
    logger.info("\n" + "=" * 50)
    logger.info("Testing on Test Set")
    logger.info("=" * 50)

    test_results = {}

    # LESR Strategy
    logger.info("\n--- LESR Strategy ---")
    lesl_strategy = LESRStrategy(
        ticker=best_config['ticker'],
        revise_state_func=best_config['trainer'].revise_state,
        trained_dqn=best_config['trainer'].dqn
    )

    framework = FINSABERFrameworkHelper(
        initial_cash=config['backtest'].get('initial_cash', 100000),
        commission_per_share=config['backtest'].get('commission', 0.001),
        min_commission=config['backtest'].get('min_commission', 0.99)
    )

    framework.load_backtest_data_single_ticker(
        data_loader,
        best_config['ticker'],
        start_date=pd.to_datetime(lesl_config['test_period'][0]),
        end_date=pd.to_datetime(lesl_config['test_period'][1])
    )

    framework.run(lesl_strategy)
    lesl_metrics = framework.evaluate(lesl_strategy)

    logger.info(f"\nLESR Results on Test Set:")
    logger.info(f"  Sharpe: {lesl_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Max DD: {lesl_metrics['max_drawdown']:.2f}%")
    logger.info(f"  Total Return: {lesl_metrics['total_return']*100:.2f}%")
    logger.info(f"  Final Value: ${lesl_metrics['final_value']:.2f}")

    test_results['lesr'] = lesl_metrics

    # Baseline Strategy
    logger.info("\n--- Baseline Strategy ---")
    baseline_trainer, baseline_val = train_baseline_dqn(
        ticker=best_config['ticker'],
        data_loader=data_loader,
        train_period=lesl_config['train_period'],
        val_period=lesl_config['val_period']
    )

    baseline_strategy = BaselineDQNStrategy(
        ticker=best_config['ticker'],
        trained_dqn=baseline_trainer.dqn
    )

    framework.reset()
    framework.load_backtest_data_single_ticker(
        data_loader,
        best_config['ticker'],
        start_date=pd.to_datetime(lesl_config['test_period'][0]),
        end_date=pd.to_datetime(lesl_config['test_period'][1])
    )

    framework.run(baseline_strategy)
    baseline_metrics = framework.evaluate(baseline_strategy)

    logger.info(f"\nBaseline Results on Test Set:")
    logger.info(f"  Sharpe: {baseline_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Max DD: {baseline_metrics['max_drawdown']:.2f}%")
    logger.info(f"  Total Return: {baseline_metrics['total_return']*100:.2f}%")
    logger.info(f"  Final Value: ${baseline_metrics['final_value']:.2f}")

    test_results['baseline'] = baseline_metrics

    # Final comparison
    logger.info("\n" + "=" * 50)
    logger.info("Final Comparison")
    logger.info("=" * 50)

    sharpe_improvement = (
        (lesl_metrics['sharpe_ratio'] / baseline_metrics['sharpe_ratio'] - 1) * 100
        if baseline_metrics['sharpe_ratio'] > 0 else 0
    )

    logger.info(f"LESR Sharpe:     {lesl_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Baseline Sharpe: {baseline_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Improvement:     {sharpe_improvement:+.1f}%")

    logger.info(f"\nLESR Max DD:     {lesl_metrics['max_drawdown']:.2f}%")
    logger.info(f"Baseline Max DD: {baseline_metrics['max_drawdown']:.2f}%")

    logger.info(f"\nLESR Return:     {lesl_metrics['total_return']*100:.2f}%")
    logger.info(f"Baseline Return: {baseline_metrics['total_return']*100:.2f}%")

    # Save results
    results = {
        'lesr': lesl_metrics,
        'baseline': baseline_metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }

    results_file = os.path.join(output_dir, 'final_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"\nResults saved to {results_file}")

    # Save summary
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Exp4.7 Financial Trading Experiment - Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Ticker: {best_config['ticker']}\n\n")

        f.write("LESR Strategy:\n")
        f.write(f"  Sharpe: {lesl_metrics['sharpe_ratio']:.3f}\n")
        f.write(f"  Max DD: {lesl_metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Total Return: {lesl_metrics['total_return']*100:.2f}%\n\n")

        f.write("Baseline Strategy:\n")
        f.write(f"  Sharpe: {baseline_metrics['sharpe_ratio']:.3f}\n")
        f.write(f"  Max DD: {baseline_metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Total Return: {baseline_metrics['total_return']*100:.2f}%\n\n")

        f.write(f"Sharpe Improvement: {sharpe_improvement:+.1f}%\n")

    logger.info(f"Summary saved to {summary_file}")


def load_best_config(results_dir: str) -> dict:
    """Load best configuration from existing results."""
    best_sharpe = -float('inf')
    best_config = None

    # Search all iteration directories
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
