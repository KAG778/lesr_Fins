"""
LESR Controller Module for exp4.9_d

Changes from exp4.7:
- D1: revise_state is fixed (from feature_engine.py), LLM only generates intrinsic_reward
- D2: Prompts describe pre-computed state structure, ask only for reward function
- A1: Generates per-ticker prompts with stock profiles
- Fixed state_dim = 150 (120 raw + 30 features) + 1 position flag
"""

import os
import sys
import importlib
import pickle
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
import logging
import torch
from torch.multiprocessing import Pool, set_start_method

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dqn_trainer import DQNTrainer
from feature_engine import revise_state as fixed_revise_state, STATE_DIM, ORIGINAL_DIM
from feature_analyzer import analyze_features
from prompts import (
    get_initial_prompt,
    generate_stock_profile,
    get_financial_cot_prompt,
    get_iteration_prompt
)

logger = logging.getLogger(__name__)


def _train_ticker_worker(args):
    """Worker for parallel training. revise_state is always from feature_engine."""
    ticker, reward_code, gpu_id, data_pkl_path, \
        train_start, train_end, val_start, val_end, sample_id, max_episodes, \
        intrinsic_weight, commission = args

    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(gpu_id) if torch.cuda.is_available() else None

        # Import reward function from code string
        import importlib.util
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        tmp.write(reward_code)
        tmp.close()
        spec = importlib.util.spec_from_file_location(f'worker_{ticker}_{gpu_id}', tmp.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.unlink(tmp.name)

        # Load data
        sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))
        from backtest.data_util.finmem_dataset import FinMemDataset
        from dqn_trainer import DQNTrainer
        from feature_engine import revise_state, STATE_DIM

        data_loader = FinMemDataset(pickle_file=data_pkl_path)

        trainer = DQNTrainer(
            ticker=ticker,
            revise_state_func=revise_state,  # Fixed from feature_engine
            intrinsic_reward_func=module.intrinsic_reward,
            state_dim=STATE_DIM,  # 150 (DQNTrainer adds +1 for position flag)
            intrinsic_weight=intrinsic_weight,
            commission=commission,
            device=device
        )

        trainer.train(data_loader, train_start, train_end, max_episodes=max_episodes)
        val_metrics = trainer.evaluate(data_loader, val_start, val_end)

        return {
            'sample_id': sample_id,
            'ticker': ticker,
            'sharpe': val_metrics['sharpe'],
            'max_dd': val_metrics['max_dd'],
            'total_return': val_metrics['total_return'],
            'num_trades': val_metrics.get('num_trades', 0),
            'states': trainer.episode_states[:100],
            'rewards': trainer.episode_rewards[:100],
            'dqn_state': trainer.get_state_dict_portable(),
            'error': None
        }
    except Exception as e:
        return {
            'sample_id': sample_id,
            'ticker': ticker,
            'sharpe': 0.0, 'max_dd': 100.0, 'total_return': 0.0, 'num_trades': 0,
            'states': [], 'rewards': [], 'dqn_state': None,
            'error': str(e)
        }


class LESRController:
    """
    Main controller for exp4.9_d.

    Key difference from exp4.7: revise_state is fixed, LLM only designs intrinsic_reward.
    """

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

        # LLM
        self.openai_key = config['openai_key']
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.base_url = config.get('base_url', None)

        # Training params
        self.intrinsic_weight = config.get('intrinsic_weight', 0.1)
        self.commission = config.get('commission', 0.001)

        # Output
        self.output_dir = config.get('output_dir', 'exp4.9_d/results')
        os.makedirs(self.output_dir, exist_ok=True)

        self.data_pkl_path = config.get('data_pkl_path', None)

        # A1: Pre-generate stock profiles
        self.stock_profiles: Dict[str, str] = {}
        self._generate_stock_profiles()

        # History
        self.all_iter_results: List[List[Dict]] = []
        self.all_iter_cot_suggestions: List[str] = []
        self.all_codes: List[List[str]] = []

        # OpenAI client
        from openai import OpenAI
        self.openai_client = OpenAI(
            api_key=self.openai_key,
            base_url=self.base_url if self.base_url else None
        )
        if self.base_url:
            logger.info(f"Using custom base_url: {self.base_url}")

    def _generate_stock_profiles(self):
        """A1: Generate stock profiles from training data."""
        for ticker in self.tickers:
            try:
                self.stock_profiles[ticker] = generate_stock_profile(
                    ticker, self.data_loader,
                    self.train_period[0], self.train_period[1]
                )
            except Exception as e:
                logger.warning(f"Profile generation failed for {ticker}: {e}")
                self.stock_profiles[ticker] = f"## Target Stock: {ticker}"

    def _get_prompt(self, iteration: int, ticker: str = None) -> str:
        """Generate prompt for given iteration."""
        profile = self.stock_profiles.get(ticker, "") if ticker else ""
        if iteration == 0:
            return get_initial_prompt(ticker=ticker, stock_profile=profile)
        else:
            return get_iteration_prompt(
                self.all_codes, self.all_iter_cot_suggestions,
                ticker=ticker, stock_profile=profile
            )

    def run_optimization(self) -> Dict:
        """Main optimization loop."""
        logger.info("=" * 50)
        logger.info("Starting LESR Optimization (exp4.9_d)")
        logger.info(f"  Fixed state_dim: {STATE_DIM} + 1 position flag = {STATE_DIM + 1}")
        logger.info(f"  intrinsic_weight: {self.intrinsic_weight}")
        logger.info(f"  commission: {self.commission}")
        print(f"exp4.9_d: state_dim={STATE_DIM}, intrinsic_weight={self.intrinsic_weight}", flush=True)

        for iteration in range(self.max_iterations):
            logger.info(f"\nIteration {iteration}")
            prompt = self._get_prompt(iteration, self.tickers[0] if self.tickers else None)

            if iteration == 0:
                valid_samples = self._sample_functions_init(prompt, iteration)
            else:
                valid_samples = self._sample_functions(prompt, iteration)

            if len(valid_samples) == 0:
                logger.warning("No valid samples, skipping")
                continue

            self.all_codes.append([s['code'] for s in valid_samples])

            logger.info(f"Training {len(valid_samples)} samples...")
            print(f"Training {len(valid_samples)} samples...", flush=True)
            results = self._parallel_train(valid_samples, iteration)

            logger.info("Analyzing results...")
            print("Analyzing results...", flush=True)
            analysis = self._analyze_results(valid_samples, results)

            logger.info("Generating COT feedback...")
            print("Generating COT feedback...", flush=True)
            cot = self._generate_cot_feedback(valid_samples, results, analysis)
            self.all_iter_cot_suggestions.append(cot)

            self._save_iteration_results(iteration, valid_samples, results, analysis)
            logger.info(f"Iteration {iteration} completed")

        return self._select_best_strategy()

    def _sample_functions_init(self, prompt, iteration):
        """Init sampling — LLM only generates intrinsic_reward."""
        all_valid = []
        counter = 0

        for round_idx in range(self.init_max_rounds):
            round_valid = []
            for _ in range(self.sample_count):
                counter += 1
                logger.info(f"[Init] Round {round_idx+1}, Sample {counter}: calling LLM...")
                print(f"[Init] Round {round_idx+1}, Sample {counter}...", flush=True)
                try:
                    resp = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a financial quantitative analysis expert."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=1500  # Only reward function, shorter
                    )
                    code = self._extract_python_code(resp.choices[0].message.content)
                    fn = f'it{iteration}_sample{len(all_valid)+len(round_valid)}.py'
                    path = os.path.join(self.output_dir, fn)
                    with open(path, 'w') as f:
                        f.write(code)

                    module = self._validate_code(path)
                    if module is not None:
                        round_valid.append({'code': code, 'module': module})
                        logger.info(f"  Sample {counter} validated")
                    else:
                        logger.warning(f"  Sample {counter} validation failed")
                except Exception as e:
                    logger.error(f"  Sample {counter} failed: {e}")

            all_valid.extend(round_valid)
            print(f"[Init] Round {round_idx+1}: {len(round_valid)} valid, "
                  f"total {len(all_valid)}/{self.init_min_valid}", flush=True)
            if len(all_valid) >= self.init_min_valid:
                break

        return all_valid

    def _sample_functions(self, prompt, iteration):
        """Sample reward functions from LLM."""
        valid = []
        for sid in range(self.sample_count):
            logger.info(f"Sample {sid+1}/{self.sample_count}...")
            print(f"Sample {sid+1}/{self.sample_count}...", flush=True)
            try:
                resp = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial quantitative analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1500
                )
                code = self._extract_python_code(resp.choices[0].message.content)
                fn = f'it{iteration}_sample{sid}.py'
                path = os.path.join(self.output_dir, fn)
                with open(path, 'w') as f:
                    f.write(code)

                module = self._validate_code(path)
                if module is not None:
                    valid.append({'code': code, 'module': module})
                    logger.info(f"  Sample {sid+1} validated")
                else:
                    logger.warning(f"  Sample {sid+1} failed")
            except Exception as e:
                logger.error(f"  Sample {sid+1} error: {e}")
        return valid

    def _extract_python_code(self, text):
        """Extract Python code from LLM output."""
        if 'import numpy as np' in text:
            start = text.index('import numpy as np')
        elif 'import numpy' in text:
            start = text.index('import numpy')
        else:
            raise ValueError("No numpy import found")

        end_marker = text.find('```', start)
        if end_marker != -1 and end_marker > start:
            section = text[start:end_marker]
            if 'return' in section:
                end = section.rindex('return')
                while end < len(section) and section[end] != '\n':
                    end += 1
                return text[start:start+end]
            return text[start:end_marker]
        else:
            end = text.rindex('return')
            while end < len(text) and text[end] != '\n':
                end += 1
            return text[start:end+1]

    def _validate_code(self, code_path):
        """Validate intrinsic_reward function only (revise_state is fixed)."""
        try:
            module_name = code_path.replace('/', '.').replace('.py', '')
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Test with 151-dim state (150 features + 1 position flag)
            test_state = np.zeros(STATE_DIM + 1)
            test_state[150] = 0.0  # position flag
            intrinsic_r = module.intrinsic_reward(test_state)

            assert -100 <= intrinsic_r <= 100, f"intrinsic_reward must be in [-100,100], got {intrinsic_r}"

            # Also test with position = 1
            test_state[150] = 1.0
            intrinsic_r2 = module.intrinsic_reward(test_state)
            assert -100 <= intrinsic_r2 <= 100, f"intrinsic_reward(pos=1) out of range: {intrinsic_r2}"

            return module
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return None

    def _parallel_train(self, samples, iteration):
        """Train DQN for each reward function sample."""
        results = []
        num_gpus = torch.cuda.device_count()

        for i, sample in enumerate(samples):
            logger.info(f"\nTraining sample {i+1}/{len(samples)}...")

            if num_gpus >= len(self.tickers):
                try:
                    set_start_method('spawn', force=True)
                except RuntimeError:
                    pass

                tasks = [(
                    ticker, sample['code'],
                    gpu_id, self.data_pkl_path,
                    self.train_period[0], self.train_period[1],
                    self.val_period[0], self.val_period[1],
                    i, 50, self.intrinsic_weight, self.commission
                ) for gpu_id, ticker in enumerate(self.tickers)]

                with Pool(processes=len(self.tickers)) as pool:
                    worker_results = pool.map(_train_ticker_worker, tasks)

                for wr in worker_results:
                    if wr['error']:
                        logger.error(f"  [{wr['ticker']}] Failed: {wr['error']}")
                        continue

                    # Reconstruct trainer for analysis
                    trainer = DQNTrainer(
                        ticker=wr['ticker'],
                        revise_state_func=fixed_revise_state,
                        intrinsic_reward_func=sample['module'].intrinsic_reward,
                        state_dim=STATE_DIM,
                        intrinsic_weight=self.intrinsic_weight,
                        commission=self.commission
                    )
                    if wr['dqn_state'] is not None:
                        trainer.load_state_dict_portable(wr['dqn_state'])
                    trainer.episode_states = wr['states']
                    trainer.episode_rewards = wr['rewards']

                    results.append({
                        'sample_id': i, 'ticker': wr['ticker'],
                        'sharpe': wr['sharpe'], 'max_dd': wr['max_dd'],
                        'total_return': wr['total_return'],
                        'num_trades': wr['num_trades'],
                        'trainer': trainer
                    })
                    logger.info(f"  [{wr['ticker']}] Sharpe={wr['sharpe']:.3f} MaxDD={wr['max_dd']:.2f}% Trades={wr['num_trades']}")
            else:
                for ticker in self.tickers:
                    try:
                        device = f'cuda:0' if num_gpus > 0 else 'cpu'
                        trainer = DQNTrainer(
                            ticker=ticker,
                            revise_state_func=fixed_revise_state,
                            intrinsic_reward_func=sample['module'].intrinsic_reward,
                            state_dim=STATE_DIM,
                            intrinsic_weight=self.intrinsic_weight,
                            commission=self.commission,
                            device=device
                        )
                        trainer.train(self.data_loader, self.train_period[0],
                                     self.train_period[1], max_episodes=50)
                        val = trainer.evaluate(self.data_loader,
                                              self.val_period[0], self.val_period[1])
                        results.append({
                            'sample_id': i, 'ticker': ticker,
                            'sharpe': val['sharpe'], 'max_dd': val['max_dd'],
                            'total_return': val['total_return'],
                            'num_trades': val.get('num_trades', 0),
                            'trainer': trainer
                        })
                        logger.info(f"  [{ticker}] Sharpe={val['sharpe']:.3f}")
                    except Exception as e:
                        logger.error(f"  [{ticker}] Failed: {e}")

        return results

    def _analyze_results(self, samples, results):
        """Analyze training results."""
        analysis = []
        for i in range(len(samples)):
            sample_results = [r for r in results if r['sample_id'] == i]
            if not sample_results:
                continue

            all_states, all_rewards = [], []
            for r in sample_results:
                s = r['trainer']._get_training_summary()
                all_states.extend(s['states'])
                all_rewards.extend(s['rewards'])

            if all_states:
                importance, correlations, shap_values = analyze_features(
                    all_states, all_rewards, ORIGINAL_DIM
                )
                analysis.append({
                    'sample_id': i,
                    'importance': importance,
                    'correlations': correlations,
                    'shap_values': shap_values
                })
        return analysis

    def _generate_cot_feedback(self, samples, results, analysis):
        """Generate COT feedback."""
        codes = [s['code'] for s in samples]
        scores = []
        for i in range(len(samples)):
            sr = [r for r in results if r['sample_id'] == i]
            if sr:
                scores.append({
                    'sharpe': np.mean([r['sharpe'] for r in sr]),
                    'max_dd': np.mean([r['max_dd'] for r in sr]),
                    'total_return': np.mean([r['total_return'] for r in sr]),
                    'num_trades': int(np.mean([r.get('num_trades', 0) for r in sr]))
                })
            else:
                scores.append({'sharpe': 0, 'max_dd': 100, 'total_return': 0, 'num_trades': 0})

        imp = [a['importance'] for a in analysis]
        corr = [a['correlations'] for a in analysis]
        return get_financial_cot_prompt(codes, scores, imp, corr, ORIGINAL_DIM)

    def _save_iteration_results(self, iteration, samples, results, analysis):
        """Save iteration results."""
        d = os.path.join(self.output_dir, f'iteration_{iteration}')
        os.makedirs(d, exist_ok=True)

        clean_samples = [{'code': s['code']} for s in samples]
        clean_results = [{
            'sample_id': r['sample_id'], 'ticker': r['ticker'],
            'sharpe': r['sharpe'], 'max_dd': r['max_dd'],
            'total_return': r['total_return'], 'num_trades': r.get('num_trades', 0)
        } for r in results]

        with open(os.path.join(d, 'results.pkl'), 'wb') as f:
            pickle.dump({'samples': clean_samples, 'results': clean_results, 'analysis': analysis}, f)

        if iteration < len(self.all_iter_cot_suggestions):
            with open(os.path.join(d, 'cot_feedback.txt'), 'w') as f:
                f.write(self.all_iter_cot_suggestions[iteration])

    def _select_best_strategy(self):
        """Select best strategy across iterations."""
        best_sharpe = -float('inf')
        best = None
        for it in range(self.max_iterations):
            rf = os.path.join(self.output_dir, f'iteration_{it}', 'results.pkl')
            if os.path.exists(rf):
                with open(rf, 'rb') as f:
                    data = pickle.load(f)
                for r in data['results']:
                    if r['sharpe'] > best_sharpe:
                        best_sharpe = r['sharpe']
                        best = {'iteration': it, 'sample_id': r['sample_id'],
                                'ticker': r['ticker'], 'sharpe': r['sharpe']}
        if best:
            logger.info(f"Best: It{best['iteration']} S{best['sample_id']} {best['ticker']} Sharpe={best['sharpe']:.3f}")
        return best
