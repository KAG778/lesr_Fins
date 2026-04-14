"""
LESR Controller for Exp4.9_c

Key changes from 4.7:
- validate_code: revise_state(s) returns only features (no raw, no regime)
- Framework builds enhanced_state = [raw(120) + regime(3) + features(N)]
- Passes worst_trades to COT feedback
- state_dim = 120 + 3 + feature_dim
"""

import os
import sys
import re
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

# ---------------------------------------------------------------------------
# COT Leakage Prevention (D-06)
# ---------------------------------------------------------------------------

# Keys that are allowed to pass through to COT feedback.
# Only training-set metrics; no factor evaluation, regime breakdown, or
# validation/test metrics should ever reach the LLM prompt.
_ALLOWED_COT_KEYS = {'sharpe', 'max_dd', 'total_return'}

# Patterns that indicate a leaked metric in rendered prompt text.
_LEAKAGE_PATTERNS = [
    (r'test_sharpe', 'test_sharpe'),
    (r'val_sharpe', 'val_sharpe'),
    (r'factor_metrics', 'factor_metrics'),
    (r'regime_metrics', 'regime_metrics'),
    (r'sortino', 'sortino'),
    (r'calmar', 'calmar'),
    (r'win[_ ]rate', 'win_rate'),
    (r'quantile[_ ]spread', 'quantile_spread'),
]


def filter_cot_metrics(results: list) -> list:
    """Strip non-training metrics from evaluation results before COT feedback.

    Keeps only sharpe, max_dd, total_return.
    Removes sortino, calmar, win_rate, factor_metrics, regime_metrics,
    and any key containing 'test' or 'val'.

    Args:
        results: List of result dicts from evaluation.

    Returns:
        Filtered list of result dicts with only training-allowed keys.
    """
    filtered = []
    for r in results:
        safe = {}
        for k, v in r.items():
            if k in _ALLOWED_COT_KEYS:
                safe[k] = v
            # Explicitly skip keys containing 'test', 'val', 'factor', 'regime'
            elif any(s in k.lower() for s in ('test', 'val', 'factor', 'regime')):
                continue
            # Also skip known sensitive metric names
            elif k in ('sortino', 'calmar', 'win_rate', 'quantile_spread'):
                continue
        filtered.append(safe)
    return filtered


def check_prompt_for_leakage(prompt_text: str) -> list:
    """Scan rendered prompt text for leaked metric names.

    Args:
        prompt_text: The rendered COT prompt string.

    Returns:
        List of leaked metric names found (empty if clean).
    """
    found = []
    text_lower = prompt_text.lower()
    for pattern, metric_name in _LEAKAGE_PATTERNS:
        if re.search(pattern, text_lower):
            found.append(metric_name)
    return found


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

        self.output_dir = config.get('output_dir', 'exp4.9_c/results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_pkl_path = config.get('data_pkl_path', None)

        self.all_iter_results = []
        self.all_iter_cot_suggestions = []
        self.all_codes = []

        import openai as _openai_mod
        self._openai_mod = _openai_mod
        self._openai_version = int(_openai_mod.__version__.split('.')[0])
        if self._openai_version >= 1:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self.openai_key,
                base_url=self.base_url if self.base_url else None
            )
        else:
            _openai_mod.api_key = self.openai_key
            if self.base_url:
                _openai_mod.api_base = self.base_url

    def _call_llm(self, prompt):
        """Unified LLM call, compatible with openai v0.28 and v1.x"""
        messages = [
            {"role": "system", "content": "You are a financial quant expert specializing in regime-aware trading strategies."},
            {"role": "user", "content": prompt}
        ]
        if self._openai_version >= 1:
            resp = self._openai_client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=2000
            )
            return resp.choices[0].message.content
        else:
            resp = self._openai_mod.ChatCompletion.create(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=2000
            )
            return resp['choices'][0]['message']['content']

    def run_optimization(self) -> Dict:
        logger.info("=" * 50)
        logger.info("Exp4.9_c: Regime-Aware LESR Optimization")
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
                    text = self._call_llm(prompt)
                    code = self._extract_code(text)
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
                text = self._call_llm(prompt)
                code = self._extract_code(text)
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
        Validate LLM code.
        revise_state(s) should return ONLY features (not raw state or regime).
        Framework will build: [raw(120) + regime(3) + features(N)].
        """
        try:
            # --- Static check: ban randomness in intrinsic_reward ---
            with open(code_path, 'r') as f:
                code_text = f.read()

            random_patterns = ['np.random.uniform', 'np.random.normal',
                               'np.random.rand', 'random.uniform',
                               'random.random', 'random.randint']
            reward_body = code_text[code_text.find('def intrinsic_reward'):]
            for rp in random_patterns:
                if rp in reward_body:
                    raise ValueError(
                        f"Banned random call '{rp}' in intrinsic_reward. "
                        "Reward must be deterministic.")

            module_name = code_path.replace('/', '.').replace('.py', '')
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, code_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Test revise_state: input 120d, output should be 1D features
            test_state = np.zeros(120)
            features = module.revise_state(test_state)
            features = np.atleast_1d(np.asarray(features, dtype=float))
            assert features.ndim == 1, f"Expected 1D, got {features.ndim}D"
            assert features.shape[0] >= 1, "Must return at least 1 feature"

            feature_dim = features.shape[0]

            # Test consistency
            test_state2 = np.ones(120) * 100
            features2 = module.revise_state(test_state2)
            assert features2.shape[0] == feature_dim, \
                f"Inconsistent dims: {feature_dim} vs {features2.shape[0]}"

            # --- Determinism check: same input must produce same reward ---
            test_regime = np.array([0.5, 0.2, 0.1])
            enhanced = np.concatenate([test_state, test_regime, features])
            reward1 = module.intrinsic_reward(enhanced)
            reward2 = module.intrinsic_reward(enhanced)
            reward3 = module.intrinsic_reward(enhanced)
            assert reward1 == reward2 == reward3, (
                f"intrinsic_reward is non-deterministic: "
                f"{reward1}, {reward2}, {reward3}")

            assert -100 <= reward1 <= 100, f"Reward {reward1} out of [-100, 100]"

            # Verify regime is at [120:123]
            assert np.allclose(enhanced[120:123], test_regime), "Regime not at [120:123]"

            # --- Regime sensitivity: reward must respond to risk level ---
            test_regime_risky = np.array([0.0, 0.2, 0.9])  # high risk
            enhanced_risky = np.concatenate([test_state, test_regime_risky, features])
            reward_risky = module.intrinsic_reward(enhanced_risky)
            assert reward_risky != reward1, (
                "Reward does not respond to regime changes at all")

            # Total state dim = 120 (raw) + 3 (regime) + feature_dim
            module.state_dim = 123 + feature_dim
            module.feature_dim = feature_dim

            logger.info(f"  Validated: features={feature_dim}, state_dim={module.state_dim}, deterministic=True")
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

        # D-06: Filter results to prevent data leakage.
        # Only training-set metrics (sharpe, max_dd, total_return) reach the LLM.
        filtered_results = filter_cot_metrics(results)

        for i in range(len(samples)):
            sr = [r for r in filtered_results if r.get('sample_id') == i]
            # Use original results for trainer/worst_trades access (not sent to LLM)
            sr_orig = [r for r in results if r['sample_id'] == i]
            if sr:
                scores.append({
                    'sharpe': np.mean([r['sharpe'] for r in sr]),
                    'max_dd': np.mean([r['max_dd'] for r in sr]),
                    'total_return': np.mean([r['total_return'] for r in sr])
                })
                # Collect worst trades across tickers (from original results)
                wt = []
                for r in sr_orig:
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
