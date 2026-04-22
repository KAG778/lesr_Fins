#!/usr/bin/env python3
"""
Run baseline DQN (no feature engineering, no LLM) for comparison with LESR.

Usage:
    python3 scripts/run_baseline.py --config configs/config_W1.yaml
    python3 scripts/run_baseline.py --all   # run all W1-W10
"""
import os, sys, json, argparse, logging
import numpy as np
import torch
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)
sys.path.insert(0, os.path.join(project_dir, 'core'))
sys.path.insert(0, os.path.join(project_dir, '..', 'FINSABER'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def zero_features(raw_state):
    """Return zero-length feature vector — baseline uses no LLM features."""
    return np.array([], dtype=float)


def run_baseline(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tickers = config['experiment']['tickers']
    train_period = config['experiment']['train_period']
    val_period = config['experiment']['val_period']
    test_period = config['experiment']['test_period']

    # Resolve data path
    pkl = config['data']['pickle_file']
    if not os.path.isabs(pkl):
        pkl = os.path.join(project_dir, pkl)

    from backtest.data_util.finmem_dataset import FinMemDataset
    from dqn_trainer import DQNTrainer

    data_loader = FinMemDataset(pickle_file=pkl)
    window_name = os.path.basename(config_path).replace('config_', '').replace('.yaml', '')

    results_dir = os.path.join(project_dir, 'results', window_name, 'baseline')
    os.makedirs(results_dir, exist_ok=True)

    # Baseline: state_dim = raw(120) + regime(3) + features(0) = 123
    STATE_DIM = 123

    logger.info(f"=== Baseline DQN for {window_name} ===")
    logger.info(f"Tickers: {tickers}  state_dim={STATE_DIM}")
    logger.info(f"Train: {train_period} | Val: {val_period} | Test: {test_period}")

    all_results = {}

    for ticker in tickers:
        logger.info(f"\n--- Baseline: {ticker} ---")

        trainer = DQNTrainer(
            ticker=ticker,
            revise_state_func=zero_features,
            state_dim=STATE_DIM,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        trainer.train(data_loader, train_period[0], train_period[1], max_episodes=50)
        val_metrics = trainer.evaluate(data_loader, val_period[0], val_period[1])
        test_metrics = trainer.evaluate(data_loader, test_period[0], test_period[1])

        logger.info(f"  {ticker} Val Sharpe={val_metrics['sharpe']:.3f} "
                     f"MaxDD={val_metrics['max_dd']:.2f}% "
                     f"Return={val_metrics['total_return']*100:.2f}%")
        logger.info(f"  {ticker} Test Sharpe={test_metrics['sharpe']:.3f} "
                     f"MaxDD={test_metrics['max_dd']:.2f}% "
                     f"Return={test_metrics['total_return']*100:.2f}%")

        all_results[ticker] = {
            'val_sharpe': val_metrics['sharpe'],
            'val_max_dd': val_metrics['max_dd'],
            'val_total_return': val_metrics['total_return'],
            'test_sharpe': test_metrics['sharpe'],
            'test_max_dd': test_metrics['max_dd'],
            'test_total_return': test_metrics['total_return'],
        }

    # Summary
    val_sharpes = [r['val_sharpe'] for r in all_results.values()]
    test_sharpes = [r['test_sharpe'] for r in all_results.values()]
    avg_val_sharpe = np.mean(val_sharpes)
    avg_test_sharpe = np.mean(test_sharpes)

    logger.info(f"\n{'='*50}")
    logger.info(f"Baseline {window_name} Summary:")
    logger.info(f"Validation Set:")
    for ticker, r in all_results.items():
        logger.info(f"  {ticker}: Sharpe={r['val_sharpe']:.3f} MaxDD={r['val_max_dd']:.2f}%")
    logger.info(f"  AVG Val Sharpe: {avg_val_sharpe:.3f}")
    logger.info(f"\nTest Set:")
    for ticker, r in all_results.items():
        logger.info(f"  {ticker}: Sharpe={r['test_sharpe']:.3f} MaxDD={r['test_max_dd']:.2f}%")
    logger.info(f"  AVG Test Sharpe: {avg_test_sharpe:.3f}")

    # Save
    summary = {
        'window': window_name,
        'avg_val_sharpe': avg_val_sharpe,
        'avg_test_sharpe': avg_test_sharpe,
        'tickers': all_results
    }
    with open(os.path.join(results_dir, 'baseline_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help='Single config YAML')
    parser.add_argument('--all', action='store_true', help='Run W1-W10')
    args = parser.parse_args()

    if args.all:
        for w in range(1, 11):
            cfg = os.path.join(project_dir, 'configs', f'config_W{w}.yaml')
            if os.path.exists(cfg):
                try:
                    run_baseline(cfg)
                except Exception as e:
                    logger.error(f"W{w} failed: {e}")
    elif args.config:
        cfg = args.config
        if not os.path.isabs(cfg):
            cfg = os.path.join(project_dir, cfg)
        run_baseline(cfg)
    else:
        logger.error("Specify --config or --all")


if __name__ == '__main__':
    main()
