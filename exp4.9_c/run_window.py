#!/usr/bin/env python3
"""
Exp4.9_c: Run one window with regime-aware LESR.
Usage: python run_window.py --config exp4.9_c/config_W1.yaml
"""
import os, sys, pickle, logging, argparse, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR.parent / 'FINSABER'))

os.makedirs(SCRIPT_DIR / 'logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

import pandas as pd, yaml, numpy as np, torch
from torch.multiprocessing import Pool
from regime_detector import detect_regime


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
    }
    controller = LESRController(lesr_config)
    return controller.run_optimization()


def _eval_worker(args):
    """Worker for test evaluation. Framework builds enhanced state."""
    ticker, gpu_id, data_pkl_path, train_start, train_end, val_start, val_end, test_start, test_end, results_dir, max_episodes = args
    try:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)

        # Find best result
        best_result = None
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

        # Validate and get dims
        test_state = np.zeros(120)
        features = module.revise_state(test_state)
        feature_dim = features.shape[0]
        state_dim = 123 + feature_dim  # 120 raw + 3 regime + features

        from backtest.data_util.finmem_dataset import FinMemDataset
        from dqn_trainer import DQNTrainer
        data_loader = FinMemDataset(pickle_file=data_pkl_path)

        # LESR
        lesr = DQNTrainer(ticker, module.revise_state, module.intrinsic_reward, state_dim, device=device)
        lesr.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        lesr_test = lesr.evaluate(data_loader, test_start, test_end)

        # Baseline (identity features, no intrinsic reward, no regime bonus)
        def baseline_revise(s):
            return np.zeros(1)
        baseline_state_dim = 120 + 3 + 1  # raw(120) + regime(3) + 1 dummy feature
        base = DQNTrainer(ticker, baseline_revise, lambda s: 0.0, baseline_state_dim, device=device)
        base.intrinsic_weight = 0.0
        base.regime_bonus_weight = 0.0
        base.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        base_test = base.evaluate(data_loader, test_start, test_end)

        return {
            'ticker': ticker, 'best_iteration': best_iteration,
            'best_sample_id': best_sample_id,
            'validation_sharpe': float(best_result['sharpe']),
            'lesr_test': lesr_test, 'baseline_test': base_test,
            'state_dim': state_dim, 'feature_dim': feature_dim,
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
    num_gpus = torch.cuda.device_count()

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    tasks = [(t, i % num_gpus, data_pkl, train_s, train_e, val_s, val_e,
              test_s, test_e, results_dir, max_ep) for i, t in enumerate(tickers)]

    with Pool(processes=len(tickers)) as pool:
        results = pool.map(_eval_worker, tasks)

    all_results = {}
    for r in results:
        if r.get('error'):
            logger.error(f"[{r['ticker']}] {r['error']}")
            continue
        all_results[r['ticker']] = r
        logger.info(f"[{r['ticker']}] LESR={r['lesr_test']['sharpe']:.3f} Base={r['baseline_test']['sharpe']:.3f}")

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
        f"# Exp4.9_c: Regime-Aware LESR",
        f"\n> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | {torch.cuda.device_count()} GPUs\n",
        "| 标的 | " + ", ".join(config['experiment']['tickers']) + " |",
        f"| 训练 | {tp[0]} ~ {tp[1]} |",
        f"| 验证 | {vp[0]} ~ {vp[1]} |",
        f"| 测试 | {tsp[0]} ~ {tsp[1]} |",
        "| 版本 | Exp4.9_c (3d regime + framework拼接 + 软引导) |",
        "\n## 测试集结果\n",
        "| 股票 | LESR Sharpe | Base Sharpe | LESR Return | Base Return | LESR MaxDD | Base MaxDD | 结论 |",
        "|------|------------|-------------|-------------|-------------|------------|------------|------|",
    ]
    for t in config['experiment']['tickers']:
        if t not in all_results: continue
        r = all_results[t]
        lt, bt = r['lesr_test'], r['baseline_test']
        ls, bs = float(lt['sharpe']), float(bt['sharpe'])
        verdict = "LESR✓" if ls > bs else "Base✓"
        lines.append(f"| {t} | **{ls:.3f}** | {bs:.3f} | **{lt['total_return']:.2f}%** | {bt['total_return']:.2f}% | {lt['max_dd']:.2f}% | {bt['max_dd']:.2f}% | {verdict} |")
    lines.append("")
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

    logger.info(f"=== Exp4.9_c: {window_name} ===")

    from backtest.data_util.finmem_dataset import FinMemDataset
    data_loader = FinMemDataset(pickle_file=config['data']['pickle_file'])

    results_dir = config['output']['output_dir']
    os.makedirs(results_dir, exist_ok=True)

    max_iter = config['experiment'].get('max_iterations', 3)
    if not os.path.exists(os.path.join(results_dir, f'iteration_{max_iter-1}', 'results.pkl')):
        logger.info("Running optimization...")
        run_lesr(config, data_loader, api_key)
    else:
        logger.info("Optimization done, skipping.")

    logger.info("Running test evaluation...")
    run_test_eval(config, results_dir)
    logger.info("Done!")


if __name__ == '__main__':
    main()
