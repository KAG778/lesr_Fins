"""
LESR Controller Module for Exp4.9: Regime-Conditioned LESR

Key changes from 4.7:
- validate_code tests revise_state(s, regime_vector) with 2 args
- regime_metrics passed to COT feedback
- Parallel worker passes regime info
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
from feature_analyzer import analyze_features
from prompts import (
    INITIAL_PROMPT,
    get_financial_cot_prompt,
    get_iteration_prompt
)

logger = logging.getLogger(__name__)


def _train_ticker_worker(args):
    """Worker function for parallel ticker training."""
    ticker, sample_code, state_dim, gpu_id, data_pkl_path, \
        train_start, train_end, val_start, val_end, sample_id, max_episodes = args

    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(gpu_id) if torch.cuda.is_available() else None

        import importlib.util
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        tmp.write(sample_code)
        tmp.close()
        spec = importlib.util.spec_from_file_location(f'worker_{ticker}_{gpu_id}', tmp.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.unlink(tmp.name)

        sys.path.insert(0, str(Path(__file__).parent.parent / 'FINSABER'))
        from backtest.data_util.finmem_dataset import FinMemDataset
        from dqn_trainer import DQNTrainer

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

        return {
            'sample_id': sample_id,
            'ticker': ticker,
            'sharpe': val_metrics['sharpe'],
            'max_dd': val_metrics['max_dd'],
            'total_return': val_metrics['total_return'],
            'states': trainer.episode_states[:100],
            'rewards': trainer.episode_rewards[:100],
            'regimes': trainer.episode_regimes[:100],  # NEW
            'dqn_state': trainer.get_state_dict_portable(),
            'state_dim': state_dim,
            'regime_metrics': val_metrics.get('regime_metrics', {}),  # NEW
            'error': None
        }
    except Exception as e:
        return {
            'sample_id': sample_id,
            'ticker': ticker,
            'sharpe': 0.0,
            'max_dd': 100.0,
            'total_return': 0.0,
            'states': [],
            'rewards': [],
            'regimes': [],
            'dqn_state': None,
            'state_dim': state_dim,
            'regime_metrics': {},
            'error': str(e)
        }


class LESRController:
    """
    Main controller for regime-conditioned LESR optimization.
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

        self.openai_key = config['openai_key']
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.base_url = config.get('base_url', None)

        self.output_dir = config.get('output_dir', 'exp4.9/results')
        os.makedirs(self.output_dir, exist_ok=True)

        self.data_pkl_path = config.get('data_pkl_path', None)

        self.all_iter_results: List[List[Dict]] = []
        self.all_iter_cot_suggestions: List[str] = []
        self.all_codes: List[List[str]] = []

        try:
            from openai import OpenAI
            self.openai_client = OpenAI(
                api_key=self.openai_key,
                base_url=self.base_url if self.base_url else None
            )
            if self.base_url:
                logger.info(f"Using custom base_url: {self.base_url}")
        except ImportError:
            logger.error("OpenAI module not found. Please install: pip install openai")
            raise

    def run_optimization(self) -> Dict:
        """Main optimization loop."""
        logger.info("=" * 50)
        logger.info("Starting Regime-Conditioned LESR Optimization (Exp4.9)")
        logger.info("=" * 50)

        for iteration in range(self.max_iterations):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"{'=' * 50}")

            # 1. Generate prompt
            if iteration == 0:
                prompt = INITIAL_PROMPT
                logger.info("Using initial regime-aware prompt")
                print("Using initial regime-aware prompt", flush=True)
            else:
                logger.info(f"Generating iteration prompt, history: {len(self.all_codes)} iters")
                print(f"Generating iteration prompt, history: {len(self.all_codes)} iters", flush=True)
                prompt = get_iteration_prompt(self.all_codes, self.all_iter_cot_suggestions)

            # 2. Sample functions from LLM
            if iteration == 0:
                valid_samples = self._sample_functions_init(prompt, iteration)
            else:
                valid_samples = self._sample_functions(prompt, iteration)

            if len(valid_samples) == 0:
                logger.warning("No valid samples generated, skipping iteration")
                continue

            self.all_codes.append([s['code'] for s in valid_samples])

            # 3. Train DQN for each sample
            logger.info(f"Training {len(valid_samples)} samples...")
            print(f"Training {len(valid_samples)} samples...", flush=True)
            results = self._parallel_train(valid_samples, iteration)

            # 4. Analyze results
            logger.info("Analyzing results with regime breakdown...")
            print("Analyzing results with regime breakdown...", flush=True)
            analysis = self._analyze_results(valid_samples, results)

            # 5. Generate COT feedback with regime metrics
            logger.info("Generating regime-aware COT feedback...")
            print("Generating regime-aware COT feedback...", flush=True)
            cot_suggestion = self._generate_cot_feedback(valid_samples, results, analysis)
            self.all_iter_cot_suggestions.append(cot_suggestion)

            # 6. Save iteration results
            self._save_iteration_results(iteration, valid_samples, results, analysis)

            logger.info(f"Iteration {iteration} completed")

        # 7. Select best strategy
        return self._select_best_strategy()

    def _sample_functions_init(self, prompt: str, iteration: int) -> List[Dict]:
        """Initialization sampling for Iteration 0."""
        all_valid = []
        sample_counter = 0

        for round_idx in range(self.init_max_rounds):
            round_valid = []

            for _ in range(self.sample_count):
                sample_counter += 1
                logger.info(f"[Init] Round {round_idx + 1}, Sample {sample_counter}: calling LLM...")
                print(f"[Init] Round {round_idx + 1}, Sample {sample_counter}: calling LLM...", flush=True)

                try:
                    response = self.openai_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a financial quantitative analysis expert specializing in regime-adaptive trading strategies."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=2000
                    )

                    code = response.choices[0].message.content
                    code = self._extract_python_code(code)

                    code_filename = f'it{iteration}_sample{len(all_valid) + len(round_valid)}.py'
                    code_path = os.path.join(self.output_dir, code_filename)
                    with open(code_path, 'w') as f:
                        f.write(code)

                    module = self._validate_code(code_path)

                    if module is not None:
                        round_valid.append({
                            'code': code,
                            'module': module,
                            'state_dim': module.state_dim,
                            'original_dim': 120
                        })
                        logger.info(f"  [Init] Sample {sample_counter} validated: state_dim={module.state_dim}")
                    else:
                        logger.warning(f"  [Init] Sample {sample_counter} validation failed")

                except Exception as e:
                    logger.error(f"  [Init] Sample {sample_counter} failed: {e}")
                    continue

            all_valid.extend(round_valid)
            logger.info(f"[Init] Round {round_idx + 1}: {len(round_valid)} valid, total {len(all_valid)}/{self.init_min_valid}")
            print(f"[Init] Round {round_idx + 1}: {len(round_valid)} valid, total {len(all_valid)}/{self.init_min_valid}", flush=True)

            if len(all_valid) >= self.init_min_valid:
                break

        return all_valid

    def _sample_functions(self, prompt: str, iteration: int) -> List[Dict]:
        """Sample and validate functions from LLM."""
        valid_samples = []

        for sample_id in range(self.sample_count):
            logger.info(f"Sampling {sample_id + 1}/{self.sample_count}...")
            print(f"Sample {sample_id + 1}/{self.sample_count}: calling LLM...", flush=True)

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a financial quantitative analysis expert specializing in regime-adaptive trading strategies."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2000
                )

                code = response.choices[0].message.content
                code = self._extract_python_code(code)

                code_filename = f'it{iteration}_sample{sample_id}.py'
                code_path = os.path.join(self.output_dir, code_filename)
                with open(code_path, 'w') as f:
                    f.write(code)

                module = self._validate_code(code_path)

                if module is not None:
                    valid_samples.append({
                        'code': code,
                        'module': module,
                        'state_dim': module.state_dim,
                        'original_dim': 120
                    })
                    logger.info(f"  Sample {sample_id + 1} validated: state_dim={module.state_dim}")
                else:
                    logger.warning(f"  Sample {sample_id + 1} validation failed")

            except Exception as e:
                logger.error(f"  Sample {sample_id + 1} failed: {e}")
                continue

        return valid_samples

    def _extract_python_code(self, text: str) -> str:
        """Extract Python code from LLM output."""
        if 'import numpy' in text:
            start = text.index('import numpy')
        elif 'import numpy as np' in text:
            start = text.index('import numpy as np')
        else:
            raise ValueError("No numpy import found in code")

        end_marker = text.find('```', start)
        if end_marker != -1 and end_marker > start:
            code_section = text[start:end_marker]
            if 'return' in code_section:
                end = code_section.rindex('return')
                while end < len(code_section) and code_section[end] != '\n':
                    end += 1
                code = text[start:start + end]
            else:
                code = text[start:end_marker]
        else:
            end = text.rindex('return')
            while end < len(text) and text[end] != '\n':
                end += 1
            code = text[start:end + 1]

        return code

    def _validate_code(self, code_path: str):
        """
        Validate generated code.
        KEY CHANGE: tests revise_state with 2 arguments (s, regime_vector).
        Ensures output dimension is consistent and >= 125.
        """
        try:
            module_name = code_path.replace('/', '.').replace('.py', '')

            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Test with 2 arguments: raw_state AND regime_vector
            test_state = np.zeros(120)
            test_regime = np.array([0.5, 0.3, -0.2, 0.1, 0.0])
            enhanced = module.revise_state(test_state, test_regime)
            intrinsic_r = module.intrinsic_reward(enhanced)

            # Validate dimensions
            out_dim = enhanced.shape[0]
            if out_dim < 125:
                logger.error(f"Output dim {out_dim} < 125 (120 raw + 5 regime)")
                return None

            # Verify regime_vector is embedded at [120:125]
            regime_embedded = enhanced[120:125]
            if not np.allclose(regime_embedded, test_regime, atol=1e-6):
                # LLM didn't put regime_vector at [120:125] — fix it ourselves
                logger.warning(f"regime_vector not at [120:125], auto-fixing (expected {test_regime}, got {regime_embedded})")
                # Wrap revise_state to enforce regime_vector placement
                original_revise = module.revise_state
                def fixed_revise_state(s, regime_vector):
                    result = original_revise(s, regime_vector)
                    # Ensure result has at least 125 dims
                    if result.shape[0] < 125:
                        # Missing regime_vector, append it
                        result = np.concatenate([result, regime_vector])
                    elif result.shape[0] >= 125:
                        # Overwrite [120:125] with actual regime_vector
                        result[120:125] = regime_vector
                    return result
                module.revise_state = fixed_revise_state
                # Re-test with fixed version
                enhanced = module.revise_state(test_state, test_regime)
                out_dim = enhanced.shape[0]

            assert out_dim >= 125, f"Output must be >= 125, got {out_dim}"
            assert -100 <= intrinsic_r <= 100, f"intrinsic_reward must be in [-100, 100], got {intrinsic_r}"

            # Test consistency: same input must produce same output dimension
            test_state2 = np.ones(120) * 100
            test_regime2 = np.array([-0.5, 0.8, 0.3, -0.2, 0.6])
            enhanced2 = module.revise_state(test_state2, test_regime2)
            if enhanced2.shape[0] != out_dim:
                logger.error(f"Inconsistent output dims: {out_dim} vs {enhanced2.shape[0]}")
                return None

            module.state_dim = out_dim

            logger.info(f"  Code validated: state_dim={out_dim}")
            return module

        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return None

    def _parallel_train(self, samples: List[Dict], iteration: int) -> List[Dict]:
        """Train DQN for each sample."""
        results = []
        num_gpus = torch.cuda.device_count()

        for i, sample in enumerate(samples):
            logger.info(f"\nTraining sample {i + 1}/{len(samples)}...")

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
                        logger.error(f"  [{wr['ticker']}] Training failed: {wr['error']}")
                        continue

                    from dqn_trainer import DQNTrainer
                    trainer = DQNTrainer(
                        ticker=wr['ticker'],
                        revise_state_func=sample['module'].revise_state,
                        intrinsic_reward_func=sample['module'].intrinsic_reward,
                        state_dim=wr['state_dim']
                    )
                    if wr['dqn_state'] is not None:
                        trainer.load_state_dict_portable(wr['dqn_state'])
                    trainer.episode_states = wr['states']
                    trainer.episode_rewards = wr['rewards']
                    trainer.episode_regimes = wr.get('regimes', [])

                    results.append({
                        'sample_id': i,
                        'ticker': wr['ticker'],
                        'sharpe': wr['sharpe'],
                        'max_dd': wr['max_dd'],
                        'total_return': wr['total_return'],
                        'trainer': trainer,
                        'regime_metrics': wr.get('regime_metrics', {})
                    })
                    logger.info(f"  [{wr['ticker']}] Sharpe: {wr['sharpe']:.3f}, MaxDD: {wr['max_dd']:.2f}%")
            else:
                for ticker in self.tickers:
                    try:
                        device = f'cuda:0' if num_gpus > 0 else 'cpu'
                        trainer = DQNTrainer(
                            ticker=ticker,
                            revise_state_func=sample['module'].revise_state,
                            intrinsic_reward_func=sample['module'].intrinsic_reward,
                            state_dim=sample['state_dim'],
                            device=device
                        )
                        trainer.train(self.data_loader, self.train_period[0],
                                     self.train_period[1], max_episodes=50)
                        val_metrics = trainer.evaluate(self.data_loader,
                                                      self.val_period[0],
                                                      self.val_period[1])
                        results.append({
                            'sample_id': i,
                            'ticker': ticker,
                            'sharpe': val_metrics['sharpe'],
                            'max_dd': val_metrics['max_dd'],
                            'total_return': val_metrics['total_return'],
                            'trainer': trainer,
                            'regime_metrics': val_metrics.get('regime_metrics', {})
                        })
                        logger.info(f"  [{ticker}] Sharpe: {val_metrics['sharpe']:.3f}, MaxDD: {val_metrics['max_dd']:.2f}%")
                    except Exception as e:
                        logger.error(f"  [{ticker}] Training failed: {e}")
                        continue

        return results

    def _analyze_results(self, samples: List[Dict], results: List[Dict]) -> List[Dict]:
        """Analyze training results with regime grouping."""
        analysis = []
        logger.info(f"Analyzing results: {len(samples)} samples, {len(results)} results")
        print(f"Analyzing results: {len(samples)} samples, {len(results)} results", flush=True)

        for i, sample in enumerate(samples):
            sample_results = [r for r in results if r['sample_id'] == i]

            if len(sample_results) == 0:
                continue

            all_states = []
            all_rewards = []
            all_regimes = []

            for result in sample_results:
                summary = result['trainer']._get_training_summary()
                all_states.extend(summary['states'])
                all_rewards.extend(summary['rewards'])
                all_regimes.extend(summary.get('regimes', []))

            if len(all_states) > 0:
                importance, correlations, shap_values, regime_importance = analyze_features(
                    all_states, all_rewards, sample['original_dim'],
                    regime_labels=all_regimes if all_regimes else None
                )

                analysis.append({
                    'sample_id': i,
                    'importance': importance,
                    'correlations': correlations,
                    'shap_values': shap_values,
                    'regime_importance': regime_importance  # NEW
                })

        logger.info(f"Analysis complete: {len(analysis)} samples")
        print(f"Analysis complete: {len(analysis)} samples", flush=True)
        return analysis

    def _generate_cot_feedback(
        self,
        samples: List[Dict],
        results: List[Dict],
        analysis: List[Dict]
    ) -> str:
        """Generate regime-aware COT feedback."""
        codes = [s['code'] for s in samples]
        scores = []

        # Aggregate regime metrics across tickers per sample
        all_regime_metrics = []

        for i, sample in enumerate(samples):
            sample_results = [r for r in results if r['sample_id'] == i]
            if len(sample_results) > 0:
                avg_sharpe = np.mean([r['sharpe'] for r in sample_results])
                avg_max_dd = np.mean([r['max_dd'] for r in sample_results])
                avg_return = np.mean([r['total_return'] for r in sample_results])
                scores.append({
                    'sharpe': avg_sharpe,
                    'max_dd': avg_max_dd,
                    'total_return': avg_return
                })

                # Aggregate regime metrics
                merged = {}
                for r in sample_results:
                    for regime, metrics in r.get('regime_metrics', {}).items():
                        if regime not in merged:
                            merged[regime] = {'sharpes': [], 'trades': 0}
                        merged[regime]['sharpes'].append(metrics.get('sharpe', 0))
                        merged[regime]['trades'] += metrics.get('trades', 0)
                
                regime_summary = {}
                for regime, data in merged.items():
                    regime_summary[regime] = {
                        'sharpe': np.mean(data['sharpes']),
                        'trades': data['trades']
                    }
                all_regime_metrics.append(regime_summary)
            else:
                scores.append({'sharpe': 0, 'max_dd': 100, 'total_return': 0})
                all_regime_metrics.append({})

        importance_list = [a['importance'] for a in analysis]
        correlations_list = [a['correlations'] for a in analysis]

        cot_prompt = get_financial_cot_prompt(
            codes, scores, importance_list, correlations_list, 120,
            regime_metrics=all_regime_metrics  # NEW
        )

        logger.info(f"COT feedback generated: {len(cot_prompt)} chars")
        print(f"COT feedback generated: {len(cot_prompt)} chars", flush=True)
        return cot_prompt

    def _save_iteration_results(
        self,
        iteration: int,
        samples: List[Dict],
        results: List[Dict],
        analysis: List[Dict]
    ) -> None:
        """Save iteration results."""
        iteration_dir = os.path.join(self.output_dir, f'iteration_{iteration}')
        os.makedirs(iteration_dir, exist_ok=True)

        clean_samples = [{
            'code': s['code'],
            'state_dim': s['state_dim'],
            'original_dim': s['original_dim']
        } for s in samples]

        clean_results = [{
            'sample_id': r['sample_id'],
            'ticker': r['ticker'],
            'sharpe': r['sharpe'],
            'max_dd': r['max_dd'],
            'total_return': r['total_return'],
            'regime_metrics': r.get('regime_metrics', {})
        } for r in results]

        # Clean analysis for pickling
        clean_analysis = []
        for a in analysis:
            ca = {k: v for k, v in a.items() if k != 'sample_id'}
            clean_analysis.append(ca)

        with open(os.path.join(iteration_dir, 'results.pkl'), 'wb') as f:
            pickle.dump({
                'samples': clean_samples,
                'results': clean_results,
                'analysis': clean_analysis
            }, f)

        if iteration < len(self.all_iter_cot_suggestions):
            with open(os.path.join(iteration_dir, 'cot_feedback.txt'), 'w') as f:
                f.write(self.all_iter_cot_suggestions[iteration])

    def _select_best_strategy(self) -> Dict:
        """Select best strategy from all iterations."""
        best_sharpe = -float('inf')
        best_config = None

        for iteration in range(self.max_iterations):
            iteration_dir = os.path.join(self.output_dir, f'iteration_{iteration}')
            result_file = os.path.join(iteration_dir, 'results.pkl')

            if os.path.exists(result_file):
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)

                for result in data['results']:
                    if result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_config = {
                            'iteration': iteration,
                            'sample_id': result['sample_id'],
                            'ticker': result['ticker'],
                            'sharpe': result['sharpe']
                        }

        if best_config is not None:
            logger.info(f"\nBest strategy: Iteration {best_config['iteration']}, "
                       f"Sample {best_config['sample_id']}, Sharpe = {best_config['sharpe']:.3f}")
        else:
            logger.warning("No valid strategy found")

        return best_config
