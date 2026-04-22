"""
LESR Controller for Exp4.9_f

Key changes from 4.7:
- validate_code: revise_state(s) returns only features (no raw, no regime)
- Framework builds enhanced_state = [raw(120) + regime(3) + features(N)]
- Passes worst_trades to COT feedback
- state_dim = 120 + 3 + feature_dim
"""

import os
import sys
import importlib
import pickle
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import logging
import torch
from torch.multiprocessing import Pool, set_start_method

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dqn_trainer import DQNTrainer
from prompts import INITIAL_PROMPT, get_financial_cot_prompt, get_iteration_prompt

logger = logging.getLogger(__name__)

from regime_detector import detect_regime


def _train_ticker_worker(args):
    """Worker for parallel training."""
    ticker, sample_code, state_dim, gpu_id, data_pkl_path, \
        train_start, train_end, val_start, val_end, sample_id, max_episodes = args

    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        import importlib.util, tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        tmp.write(sample_code)
        tmp.close()
        spec = importlib.util.spec_from_file_location(f'w_{ticker}_{gpu_id}', tmp.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.unlink(tmp.name)

        sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))
        from backtest.data_util.finmem_dataset import FinMemDataset

        data_loader = FinMemDataset(pickle_file=data_pkl_path)

        trainer = DQNTrainer(
            ticker=ticker,
            revise_state_func=module.revise_state,
            intrinsic_reward_func=module.intrinsic_reward,
            state_dim=state_dim,
            device=device
        )

        trainer.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        val_metrics = trainer.evaluate(data_loader, val_start, val_end)

        summary = trainer._get_summary()

        return {
            'sample_id': sample_id, 'ticker': ticker,
            'sharpe': val_metrics['sharpe'], 'max_dd': val_metrics['max_dd'],
            'total_return': val_metrics['total_return'],
            'states': summary['states'][:100],
            'rewards': summary['rewards'][:100],
            'regimes': summary['regimes'][:100],
            'worst_trades': summary.get('worst_trades', [])[:10],
            'dqn_state': trainer.get_state_dict_portable(),
            'state_dim': state_dim,
            'error': None
        }
    except Exception as e:
        import traceback
        return {
            'sample_id': sample_id, 'ticker': ticker,
            'sharpe': 0.0, 'max_dd': 100.0, 'total_return': 0.0,
            'states': [], 'rewards': [], 'regimes': [], 'worst_trades': [],
            'dqn_state': None, 'state_dim': state_dim,
            'error': f'{e}\n{traceback.format_exc()}'
        }


class LESRController:
    def __init__(self, config: Dict):
        self.tickers = config['tickers']
        self.train_period = config['train_period']
        self.val_period = config['val_period']
        self.test_period = config['test_period']
        self.data_loader = config['data_loader']
        self.sample_count = config.get('sample_count', 6)
        self.max_iterations = config.get('max_iterations', 3)
        self.init_min_valid = config.get('init_min_valid', 3)
        self.init_max_rounds = config.get('init_max_rounds', 5)

        self.openai_key = config['openai_key']
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.7)
        self.base_url = config.get('base_url', None)

        self.output_dir = config.get('output_dir', 'exp4.9_f/results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_pkl_path = config.get('data_pkl_path', None)

        self.all_iter_results = []
        self.all_iter_cot_suggestions = []
        self.all_codes = []

        from openai import OpenAI
        self.openai_client = OpenAI(
            api_key=self.openai_key,
            base_url=self.base_url if self.base_url else None
        )

    def run_optimization(self) -> Dict:
        logger.info("=" * 50)
        logger.info("Exp4.9_f: Regime-Aware LESR Optimization")
        logger.info("=" * 50)

        for iteration in range(self.max_iterations):
            logger.info(f"\n=== Iteration {iteration} ===")

            if iteration == 0:
                prompt = INITIAL_PROMPT
                logger.info("Using initial regime-aware prompt")
            else:
                prompt = get_iteration_prompt(self.all_codes, self.all_iter_cot_suggestions)

            if iteration == 0:
                valid_samples = self._sample_functions_init(prompt, iteration)
            else:
                valid_samples = self._sample_functions(prompt, iteration)

            if not valid_samples:
                logger.warning("No valid samples, skipping iteration")
                continue

            self.all_codes.append([s['code'] for s in valid_samples])

            logger.info(f"Training {len(valid_samples)} samples...")
            results = self._parallel_train(valid_samples, iteration)

            logger.info("Analyzing results...")
            analysis = self._analyze_results(valid_samples, results)

            logger.info("Generating COT feedback...")
            cot = self._generate_cot_feedback(valid_samples, results, analysis)
            self.all_iter_cot_suggestions.append(cot)

            self._save_iteration_results(iteration, valid_samples, results, analysis)

        return self._select_best_strategy()

    def _sample_functions_init(self, prompt, iteration):
        all_valid = []
        for round_idx in range(self.init_max_rounds):
            for s_idx in range(self.sample_count):
                logger.info(f"[Init] R{round_idx+1} S{s_idx+1}: calling LLM...")
                print(f"[Init] R{round_idx+1} S{s_idx+1}: calling LLM...", flush=True)
                try:
                    resp = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a financial quant expert specializing in regime-aware trading strategies."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature, max_tokens=2000
                    )
                    code = self._extract_code(resp.choices[0].message.content)
                    code_path = os.path.join(self.output_dir, f'it{iteration}_sample{len(all_valid)}.py')
                    with open(code_path, 'w') as f:
                        f.write(code)
                    module = self._validate_code(code_path)
                    if module:
                        all_valid.append({
                            'code': code, 'module': module,
                            'state_dim': module.state_dim, 'original_dim': 120
                        })
                        logger.info(f"  Validated: state_dim={module.state_dim}")
                except Exception as e:
                    logger.error(f"  Failed: {e}")
            if len(all_valid) >= self.init_min_valid:
                break
        return all_valid

    def _sample_functions(self, prompt, iteration):
        valid = []
        for s_id in range(self.sample_count):
            logger.info(f"Sample {s_id+1}/{self.sample_count}...")
            try:
                resp = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial quant expert specializing in regime-aware trading strategies."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature, max_tokens=2000
                )
                code = self._extract_code(resp.choices[0].message.content)
                code_path = os.path.join(self.output_dir, f'it{iteration}_sample{s_id}.py')
                with open(code_path, 'w') as f:
                    f.write(code)
                module = self._validate_code(code_path)
                if module:
                    valid.append({
                        'code': code, 'module': module,
                        'state_dim': module.state_dim, 'original_dim': 120
                    })
            except Exception as e:
                logger.error(f"  Failed: {e}")
        return valid

    def _extract_code(self, text):
        markers = ['import numpy', 'import numpy as np']
        start = -1
        for m in markers:
            if m in text:
                start = text.index(m)
                break
        if start == -1:
            raise ValueError("No numpy import found")
        code = text[start:]
        end = code.rfind('```')
        if end > 0:
            code = code[:end]
        return code.strip()

    def _validate_code(self, code_path):
        """
        Validate LLM code with 4 input cases (catches off-by-one bugs).
        revise_state(s) should return ONLY features (not raw state or regime).
        Framework will build: [raw(120) + regime(3) + features(N)].
        """
        try:
            module_name = code_path.replace('/', '.').replace('.py', '')
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Test with 4 diverse inputs to catch off-by-one / slicing bugs
            test_cases = [
                np.zeros(120),                          # all zeros
                np.ones(120) * 100,                     # all 100
                np.sin(np.linspace(0, 4*np.pi, 120)) * 50 + 100,  # sinusoidal
                np.random.randn(120) * 10 + 50,        # random noisy
            ]
            feature_dim = None
            for tc in test_cases:
                features = module.revise_state(tc)
                features = np.atleast_1d(np.asarray(features, dtype=float))
                assert features.ndim == 1, f"Expected 1D, got {features.ndim}D"
                assert features.shape[0] >= 1, "Must return at least 1 feature"
                assert np.all(np.isfinite(features)), f"Non-finite values: {features}"
                if feature_dim is None:
                    feature_dim = features.shape[0]
                else:
                    assert features.shape[0] == feature_dim, \
                        f"Inconsistent dims: {feature_dim} vs {features.shape[0]}"

            # Test intrinsic_reward with all 4 cases + different regimes
            for tc in test_cases:
                for regime in [np.array([0.5, 0.2, 0.1]),   # normal
                               np.array([0.8, 0.7, 0.9]),   # crisis
                               np.array([-0.5, 0.1, 0.1])]: # bearish calm
                    features = module.revise_state(tc)
                    features = np.atleast_1d(np.asarray(features, dtype=float))
                    enhanced = np.concatenate([tc, regime, features])
                    assert enhanced.shape[0] == 123 + feature_dim
                    assert np.allclose(enhanced[120:123], regime), "Regime not at [120:123]"
                    reward = module.intrinsic_reward(enhanced)
                    assert np.isfinite(reward), f"Non-finite reward: {reward}"
                    assert -100 <= reward <= 100, f"Reward {reward} out of [-100, 100]"

            module.state_dim = 123 + feature_dim
            module.feature_dim = feature_dim

            logger.info(f"  Validated (4 cases): features={feature_dim}, state_dim={module.state_dim}")
            return module

        except Exception as e:
            logger.error(f"  Validation failed: {e}")
            return None

    def _parallel_train(self, samples, iteration):
        results = []
        num_gpus = torch.cuda.device_count()

        for i, sample in enumerate(samples):
            logger.info(f"Training sample {i+1}/{len(samples)}...")

            if num_gpus >= len(self.tickers):
                try:
                    set_start_method('spawn', force=True)
                except RuntimeError:
                    pass

                tasks = []
                for gpu_id, ticker in enumerate(self.tickers):
                    tasks.append((
                        ticker, sample['code'], sample['state_dim'], gpu_id,
                        self.data_pkl_path,
                        self.train_period[0], self.train_period[1],
                        self.val_period[0], self.val_period[1],
                        i, 50
                    ))

                with Pool(processes=len(self.tickers)) as pool:
                    worker_results = pool.map(_train_ticker_worker, tasks)

                for wr in worker_results:
                    if wr['error']:
                        logger.error(f"  [{wr['ticker']}] {wr['error']}")
                        continue
                    trainer = DQNTrainer(
                        ticker=wr['ticker'],
                        revise_state_func=sample['module'].revise_state,
                        intrinsic_reward_func=sample['module'].intrinsic_reward,
                        state_dim=wr['state_dim']
                    )
                    if wr['dqn_state']:
                        trainer.load_state_dict_portable(wr['dqn_state'])
                    trainer.episode_states = wr['states']
                    trainer.episode_rewards = wr['rewards']
                    trainer.episode_regimes = wr['regimes']
                    trainer.worst_trades = wr.get('worst_trades', [])

                    results.append({
                        'sample_id': i, 'ticker': wr['ticker'],
                        'sharpe': wr['sharpe'], 'max_dd': wr['max_dd'],
                        'total_return': wr['total_return'],
                        'trainer': trainer
                    })
                    logger.info(f"  [{wr['ticker']}] Sharpe={wr['sharpe']:.3f}")
            else:
                for ticker in self.tickers:
                    try:
                        device = 'cuda:0' if num_gpus > 0 else 'cpu'
                        trainer = DQNTrainer(
                            ticker=ticker,
                            revise_state_func=sample['module'].revise_state,
                            intrinsic_reward_func=sample['module'].intrinsic_reward,
                            state_dim=sample['state_dim'], device=device
                        )
                        trainer.train(self.data_loader, self.train_period[0],
                                     self.train_period[1], max_episodes=50)
                        val_metrics = trainer.evaluate(self.data_loader,
                                                      self.val_period[0], self.val_period[1])
                        results.append({
                            'sample_id': i, 'ticker': ticker,
                            'sharpe': val_metrics['sharpe'],
                            'max_dd': val_metrics['max_dd'],
                            'total_return': val_metrics['total_return'],
                            'trainer': trainer
                        })
                        logger.info(f"  [{ticker}] Sharpe={val_metrics['sharpe']:.3f}")
                    except Exception as e:
                        logger.error(f"  [{ticker}] Failed: {e}")
        return results

    def _analyze_results(self, samples, results):
        from feature_analyzer import analyze_features
        analysis = []
        for i, sample in enumerate(samples):
            sample_results = [r for r in results if r['sample_id'] == i]
            if not sample_results: continue
            all_states, all_rewards, all_regimes = [], [], []
            for r in sample_results:
                s = r['trainer']._get_summary()
                all_states.extend(s['states'])
                all_rewards.extend(s['rewards'])
                all_regimes.extend(s['regimes'])
            if all_states:
                imp, corr, shap = analyze_features(
                    np.array(all_states), np.array(all_rewards), 120
                )
                analysis.append({'sample_id': i, 'importance': imp, 'correlations': corr})
        return analysis

    def _generate_cot_feedback(self, samples, results, analysis):
        codes = [s['code'] for s in samples]
        scores, worst_trades_list = [], []

        for i in range(len(samples)):
            sr = [r for r in results if r['sample_id'] == i]
            if sr:
                scores.append({
                    'sharpe': np.mean([r['sharpe'] for r in sr]),
                    'max_dd': np.mean([r['max_dd'] for r in sr]),
                    'total_return': np.mean([r['total_return'] for r in sr])
                })
                # Collect worst trades across tickers
                wt = []
                for r in sr:
                    wt.extend(r['trainer'].worst_trades)
                wt.sort(key=lambda x: x['return'])
                worst_trades_list.append(wt[:5])
            else:
                scores.append({'sharpe': 0, 'max_dd': 100, 'total_return': 0})
                worst_trades_list.append([])

        imp_list = [a['importance'] for a in analysis]
        corr_list = [a['correlations'] for a in analysis]

        return get_financial_cot_prompt(
            codes, scores, imp_list, corr_list, 120,
            worst_trades=worst_trades_list
        )

    def _save_iteration_results(self, iteration, samples, results, analysis):
        it_dir = os.path.join(self.output_dir, f'iteration_{iteration}')
        os.makedirs(it_dir, exist_ok=True)

        clean = {
            'samples': [{'code': s['code'], 'state_dim': s['state_dim']} for s in samples],
            'results': [{'sample_id': r['sample_id'], 'ticker': r['ticker'],
                         'sharpe': r['sharpe'], 'max_dd': r['max_dd'],
                         'total_return': r['total_return']} for r in results]
        }
        with open(os.path.join(it_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(clean, f)
        if iteration < len(self.all_iter_cot_suggestions):
            with open(os.path.join(it_dir, 'cot_feedback.txt'), 'w') as f:
                f.write(self.all_iter_cot_suggestions[iteration])

    def _select_best_strategy(self):
        best_sharpe, best_cfg = -float('inf'), None
        for it in range(self.max_iterations):
            rf = os.path.join(self.output_dir, f'iteration_{it}', 'results.pkl')
            if os.path.exists(rf):
                with open(rf, 'rb') as f:
                    data = pickle.load(f)
                for r in data['results']:
                    if r['sharpe'] > best_sharpe:
                        best_sharpe = r['sharpe']
                        best_cfg = r
        if best_cfg:
            logger.info(f"Best: It{best_cfg.get('sample_id',0)}, Sharpe={best_sharpe:.3f}")
        return best_cfg
