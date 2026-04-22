"""Run 2023 test-set evaluation for the best TSLA validation sample vs baseline."""

import importlib.util
import logging
import pickle
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "FINSABER"))

from dqn_trainer import DQNTrainer
from baseline import train_baseline_dqn
from main_simple import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config():
    with open(ROOT / 'config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_iteration_results(iteration: int):
    with open(ROOT / 'results' / f'iteration_{iteration}' / 'results.pkl', 'rb') as f:
        return pickle.load(f)


def find_best_tsla_result():
    best = None
    best_iteration = None
    for iteration in range(3):
        data = load_iteration_results(iteration)
        for result in data['results']:
            if result['ticker'] != 'TSLA':
                continue
            if best is None or float(result['sharpe']) > float(best['sharpe']):
                best = result
                best_iteration = iteration
    return best_iteration, best


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    config = load_config()
    data_loader = load_data(config['data']['pickle_file'])
    train_period = tuple(config['experiment']['train_period'])
    val_period = tuple(config['experiment']['val_period'])
    test_period = tuple(config['experiment']['test_period'])

    best_iteration, best_result = find_best_tsla_result()
    best_sample_id = int(best_result['sample_id'])
    best_ticker = best_result['ticker']
    best_val_sharpe = float(best_result['sharpe'])

    iteration_data = load_iteration_results(best_iteration)
    sample = iteration_data['samples'][best_sample_id]

    code_file = ROOT / 'results' / f'it{best_iteration}_sample{best_sample_id}.py'
    module = load_module_from_file(f'tsla_best_it{best_iteration}_sample{best_sample_id}', code_file)

    # Recompute state_dim from current code instead of trusting stale saved metadata
    import numpy as np
    state_dim = len(module.revise_state(np.zeros(120)))

    logger.info('=' * 60)
    logger.info('Best TSLA validation sample')
    logger.info('iteration=%s sample=%s ticker=%s val_sharpe=%.3f state_dim=%s',
                best_iteration, best_sample_id, best_ticker, best_val_sharpe, state_dim)
    logger.info('test_period=%s to %s', test_period[0], test_period[1])
    logger.info('=' * 60)

    lesr_trainer = DQNTrainer(
        ticker=best_ticker,
        revise_state_func=module.revise_state,
        intrinsic_reward_func=module.intrinsic_reward,
        state_dim=state_dim,
    )
    logger.info('Training LESR trainer for %s on %s to %s', best_ticker, train_period[0], train_period[1])
    lesr_trainer.train(data_loader, train_period[0], train_period[1], max_episodes=config['dqn'].get('max_episodes', 50))

    logger.info('Evaluating LESR on test period')
    lesr_test = lesr_trainer.evaluate(data_loader, test_period[0], test_period[1])

    logger.info('Training baseline trainer for %s', best_ticker)
    baseline_trainer, baseline_val = train_baseline_dqn(
        ticker=best_ticker,
        data_loader=data_loader,
        train_period=train_period,
        val_period=val_period,
        state_dim=120,
        intrinsic_weight=0.0,
    )
    logger.info('Evaluating baseline on test period')
    baseline_test = baseline_trainer.evaluate(data_loader, test_period[0], test_period[1])

    results = {
        'best_iteration': best_iteration,
        'best_sample_id': best_sample_id,
        'ticker': best_ticker,
        'validation_sharpe': best_val_sharpe,
        'test_period': test_period,
        'lesr_test': lesr_test,
        'baseline_val': baseline_val,
        'baseline_test': baseline_test,
        'code_file': str(code_file),
    }

    out_file = ROOT / 'results' / 'test_set_eval_tsla_best_sample.pkl'
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)

    print('\n' + '=' * 60)
    print('2023 TSLA TEST SET RESULTS')
    print('=' * 60)
    print(f'Best validation sample: iteration={best_iteration}, sample={best_sample_id}, ticker={best_ticker}, val_sharpe={best_val_sharpe:.3f}')
    print(f'Code file: {code_file.name}')
    print()
    print('LESR on 2023 test:')
    print(f"  Sharpe: {float(lesr_test['sharpe']):.3f}")
    print(f"  MaxDD: {float(lesr_test['max_dd']):.2f}%")
    print(f"  Return: {float(lesr_test['total_return']):.2f}%")
    print()
    print('Baseline on 2023 test:')
    print(f"  Sharpe: {float(baseline_test['sharpe']):.3f}")
    print(f"  MaxDD: {float(baseline_test['max_dd']):.2f}%")
    print(f"  Return: {float(baseline_test['total_return']):.2f}%")
    print()
    if float(baseline_test['sharpe']) != 0:
        improvement = (float(lesr_test['sharpe']) / float(baseline_test['sharpe']) - 1) * 100
        print(f'Sharpe improvement vs baseline: {improvement:+.1f}%')
    else:
        print('Sharpe improvement vs baseline: baseline Sharpe is 0, ratio undefined')
    print(f'Saved detailed results to: {out_file}')


if __name__ == '__main__':
    main()
