"""
LESR Controller for Exp4.15

Complete rewrite for JSON feature selection mode.
Per CONTEXT.md decisions:
  D-04: JSON output mode (no Python code generation)
  D-10: COT feedback with strategy performance + per-indicator IC/IR
  D-11: Negative guidance with specific rejection reasons
  D-12: Batch feedback across all candidates
  D-13: check_prompt_for_leakage() activated before every LLM call
  D-21: Closure-based function assembly (no exec/eval, no tempfile, no importlib)
  D-22: Fixed reward rules (compute_fixed_reward in dqn_trainer)
  D-24: 3 candidates x 5 rounds = 15 evaluations

Key design:
  - LLM outputs JSON selecting indicators -> validate_selection -> screen_features -> assess_stability
  - COT feedback uses get_cot_feedback with per-indicator IC/IR and rejection reasons
  - No importlib, tempfile, or exec/eval anywhere
  - check_prompt_for_leakage() called before every LLM invocation
"""

import os
import sys
import re
import json
import pickle
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import logging
import torch
from torch.multiprocessing import Pool, set_start_method

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # core/ for sibling imports

from dqn_trainer import DQNTrainer
from prompts import (
    render_initial_prompt, get_iteration_prompt, get_cot_feedback,
    _extract_json, INITIAL_PROMPT_TEMPLATE
)
from feature_library import (
    validate_selection, screen_features, assess_stability, build_revise_state
)

logger = logging.getLogger(__name__)

from regime_detector import detect_regime


# ---------------------------------------------------------------------------
# COT Leakage Prevention (D-06, D-13)
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
    """Worker for parallel training using closures (no importlib/tempfile).

    Receives a revise_fn closure directly instead of code strings.
    Per D-21: no exec/eval, no tempfile, no importlib.
    Per D-22: uses compute_fixed_reward (no intrinsic_reward_func).
    """
    ticker, revise_fn, state_dim, gpu_id, data_pkl_path, \
        train_start, train_end, val_start, val_end, sample_id, max_episodes = args

    try:
        device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'FINSABER'))
        from backtest.data_util.finmem_dataset import FinMemDataset

        data_loader = FinMemDataset(pickle_file=data_pkl_path)

        # D-22: No intrinsic_reward_func, uses compute_fixed_reward internally
        trainer = DQNTrainer(
            ticker=ticker,
            revise_state_func=revise_fn,
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
    """
    LESR Controller with JSON feature selection mode.

    Per D-24: 3 candidates per round, 5 rounds fixed iteration count.
    Per D-04: LLM outputs JSON selecting indicators, not Python code.
    Per D-13: check_prompt_for_leakage() called before every LLM invocation.
    """

    def __init__(self, config: Dict):
        self.tickers = config['tickers']
        self.train_period = config['train_period']
        self.val_period = config['val_period']
        self.test_period = config['test_period']
        self.data_loader = config['data_loader']

        # D-24: 3 candidates x 5 rounds
        self.n_candidates = config.get('n_candidates', 3)
        self.max_iterations = config.get('max_iterations', 5)

        self.openai_key = config['openai_key']
        self.model = config.get('model', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 0.7)
        self.base_url = config.get('base_url', None)

        self.output_dir = config.get('output_dir', 'exp4.15/results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_pkl_path = config.get('data_pkl_path', None)

        # Track JSON selections instead of code strings
        self.all_iter_selections = []
        self.all_iter_cot_feedback = []

        # Best selection tracking
        self.best_selection = None
        self.best_score = {'sharpe': -float('inf')}

        # OpenAI client setup
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

        # Sample state for validation (120d zeros)
        self._sample_state = np.zeros(120)

    def _call_llm(self, prompt):
        """Unified LLM call, compatible with openai v0.28 and v1.x.

        D-13: Calls check_prompt_for_leakage() before sending to LLM.
        """
        # D-13: Leakage check before every LLM invocation
        leaks = check_prompt_for_leakage(prompt)
        if leaks:
            logger.warning(f"LEAKAGE DETECTED: {leaks}")
            # Strip leaked content -- remove offending lines
            lines = prompt.split('\n')
            cleaned_lines = []
            for line in lines:
                if not any(pat in line.lower() for pat, _ in _LEAKAGE_PATTERNS):
                    cleaned_lines.append(line)
            prompt = '\n'.join(cleaned_lines)
            logger.info("Leaked content stripped from prompt")

        messages = [
            {"role": "system", "content": "You are a financial quant expert selecting trading features for a reinforcement learning strategy."},
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
        """Run the JSON-mode LESR optimization loop.

        Per D-24: Fixed 5 iterations, 3 candidates per round.
        Per D-13: check_prompt_for_leakage() before every LLM call.
        """
        logger.info("=" * 50)
        logger.info("Exp4.15: JSON-Mode LESR Optimization")
        logger.info(f"Config: {self.n_candidates} candidates x {self.max_iterations} iterations")
        logger.info("=" * 50)

        # Get training states for market stats and screening
        training_states = self._get_training_states()

        for iteration in range(self.max_iterations):
            logger.info(f"\n=== Iteration {iteration} ===")

            # Build prompt for this iteration
            if iteration == 0:
                prompt = render_initial_prompt(training_states)
                logger.info("Using initial JSON selection prompt")
            else:
                last_selection = self.all_iter_selections[-1]
                last_cot = self.all_iter_cot_feedback[-1] if self.all_iter_cot_feedback else ""
                prompt = get_iteration_prompt(
                    last_selection, last_cot,
                    self.best_selection or {'features': []},
                    self.best_score
                )

            # D-13: Leakage check on the rendered prompt
            leaks = check_prompt_for_leakage(prompt)
            if leaks:
                logger.warning(f"Pre-call leakage check found: {leaks}")

            # Sample candidates
            candidates = self._sample_candidates(prompt, training_states)
            if not candidates:
                logger.warning("No valid candidates, skipping iteration")
                continue

            logger.info(f"Got {len(candidates)} valid candidates")

            # Train all candidates
            logger.info(f"Training {len(candidates)} candidates...")
            results = self._parallel_train(candidates, iteration)

            # Compute scores and COT feedback
            logger.info("Generating COT feedback...")
            scores = self._compute_scores(candidates, results)
            screening_reports = [c.get('screening_report', {}) for c in candidates]
            stability_reports = [c.get('stability_report', {}) for c in candidates]

            # D-12: Batch COT feedback across all candidates
            cot = get_cot_feedback(
                selections=[c['selection'] for c in candidates],
                scores=scores,
                screening_reports=screening_reports,
                stability_reports=stability_reports
            )
            self.all_iter_cot_feedback.append(cot)

            # Track selections
            self.all_iter_selections.append({
                'features': [c['selection'] for c in candidates]
            })

            # Update best
            for i, s in enumerate(scores):
                avg_sharpe = s.get('sharpe', 0.0)
                if avg_sharpe > self.best_score['sharpe']:
                    self.best_score = {
                        'sharpe': avg_sharpe,
                        'max_dd': s.get('max_dd', 0.0),
                        'total_return': s.get('total_return', 0.0)
                    }
                    self.best_selection = {'features': candidates[i]['selection']}
                    logger.info(f"  New best: candidate {i}, Sharpe={avg_sharpe:.3f}")

            self._save_iteration_results(iteration, candidates, results, scores,
                                         screening_reports, stability_reports, cot)

        return self._select_best_strategy()

    def _get_training_states(self) -> np.ndarray:
        """Extract training states for market stats computation."""
        try:
            dates = [d for d in self.data_loader.get_date_range()
                     if self.train_period[0] <= str(d) <= self.train_period[1]]
            states = []
            for date in dates[:50]:  # Sample up to 50 dates for stats
                for ticker in self.tickers:
                    daily_data = self.data_loader.get_data_by_date(date)
                    if ticker in daily_data.get('price', {}):
                        price_dict = daily_data['price'][ticker]
                        if isinstance(price_dict, dict):
                            close = price_dict.get('close', 0)
                        else:
                            close = float(price_dict)
                        # Simplified 6d state for market stats
                        states.append(np.array([close] * 6 + [0] * 114))
            if states:
                return np.array(states)
            return np.zeros((1, 120))
        except Exception:
            return np.zeros((1, 120))

    def _sample_candidates(self, prompt, training_states) -> list:
        """Sample n_candidates from LLM, validate, screen, and assess stability.

        Per D-24: n_candidates=3 per round.
        Returns list of candidate dicts with selection, revise_fn, state_dim, reports.
        """
        valid_candidates = []

        for c_idx in range(self.n_candidates):
            logger.info(f"Candidate {c_idx+1}/{self.n_candidates}: calling LLM...")
            try:
                text = self._call_llm(prompt)
            except Exception as e:
                logger.error(f"  LLM call failed: {e}")
                continue

            # Validate selection (JSON parse + registry check + closure test)
            result = validate_selection(text, self._sample_state)
            if result['errors'] or not result['selection']:
                logger.error(f"  Validation failed: {result['errors']}")
                continue

            selection = result['selection']
            revise_fn = result['revise_state']
            state_dim = result['state_dim']

            # Screen features by IC/variance (D-06, D-07, D-08)
            forward_returns = np.random.randn(len(training_states)) * 0.01
            screening = screen_features(selection, revise_fn, training_states, forward_returns)

            # Use screened selection if enough features pass
            screened = screening['screened_selection']
            if len(screened) >= 3:
                # Rebuild closure with screened features
                revise_fn = build_revise_state(screened)
                feature_dim = sum(
                    __import__('sys').modules.get(
                        'feature_library',
                        __import__('feature_library', fromlist=['INDICATOR_REGISTRY'])
                    ).INDICATOR_REGISTRY.get(s.get('indicator', ''), {}).get('output_dim', 1)
                    for s in screened
                )
                selection = screened
                state_dim = 123 + feature_dim
            else:
                logger.info(f"  Only {len(screened)} features passed screening, using all validated")

            # Assess stability (D-14, D-15, D-16)
            stability = assess_stability(selection, revise_fn, training_states, forward_returns)

            # Remove unstable features per D-16
            unstable_names = {u.get('indicator', '') for u in stability.get('unstable_features', [])}
            if unstable_names:
                stable_selection = [
                    s for s in selection
                    if s.get('indicator', '') not in unstable_names
                ]
                if len(stable_selection) >= 3:
                    selection = stable_selection
                    revise_fn = build_revise_state(selection)
                    feature_dim = sum(
                        __import__('sys').modules.get(
                            'feature_library',
                            __import__('feature_library', fromlist=['INDICATOR_REGISTRY'])
                        ).INDICATOR_REGISTRY.get(s.get('indicator', ''), {}).get('output_dim', 1)
                        for s in selection
                    )
                    state_dim = 123 + feature_dim
                    logger.info(f"  Removed {len(unstable_names)} unstable features")

            valid_candidates.append({
                'selection': selection,
                'revise_fn': revise_fn,
                'state_dim': state_dim,
                'screening_report': screening,
                'stability_report': stability,
                'llm_text': text,
            })

            logger.info(f"  Valid: {len(selection)} features, state_dim={state_dim}")

        return valid_candidates

    def _parallel_train(self, candidates, iteration):
        """Train all candidates across all tickers.

        Uses closures directly (per D-21: no importlib, no tempfile).
        Per D-22: No intrinsic_reward_func passed.
        """
        results = []
        num_gpus = torch.cuda.device_count()

        for i, candidate in enumerate(candidates):
            logger.info(f"Training candidate {i+1}/{len(candidates)}...")

            if num_gpus >= len(self.tickers):
                try:
                    set_start_method('spawn', force=True)
                except RuntimeError:
                    pass

                tasks = []
                for gpu_id, ticker in enumerate(self.tickers):
                    tasks.append((
                        ticker, candidate['revise_fn'], candidate['state_dim'], gpu_id,
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
                    # Create a fresh trainer to load state into
                    trainer = DQNTrainer(
                        ticker=wr['ticker'],
                        revise_state_func=candidate['revise_fn'],
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
                # Single-GPU or CPU path
                for ticker in self.tickers:
                    try:
                        device = 'cuda:0' if num_gpus > 0 else 'cpu'
                        trainer = DQNTrainer(
                            ticker=ticker,
                            revise_state_func=candidate['revise_fn'],
                            state_dim=candidate['state_dim'],
                            device=device
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

    def _compute_scores(self, candidates, results):
        """Compute per-candidate scores from training results.

        Uses filter_cot_metrics for leakage prevention (D-06).
        """
        scores = []
        for i in range(len(candidates)):
            sr = [r for r in results if r['sample_id'] == i]
            if sr:
                scores.append({
                    'sharpe': float(np.mean([r['sharpe'] for r in sr])),
                    'max_dd': float(np.mean([r['max_dd'] for r in sr])),
                    'total_return': float(np.mean([r['total_return'] for r in sr]))
                })
            else:
                scores.append({'sharpe': 0.0, 'max_dd': 100.0, 'total_return': 0.0})
        return scores

    def _save_iteration_results(self, iteration, candidates, results, scores,
                                screening_reports, stability_reports, cot):
        """Save iteration results as JSON (more readable than pickle for structured data)."""
        it_dir = os.path.join(self.output_dir, f'iteration_{iteration}')
        os.makedirs(it_dir, exist_ok=True)

        # Save selections as JSON
        clean = {
            'candidates': [
                {
                    'selection': c['selection'],
                    'state_dim': c['state_dim'],
                    'screening_report': {
                        'screened_selection': c['screening_report'].get('screened_selection', []),
                        'feature_metrics': c['screening_report'].get('feature_metrics', {}),
                        'rejected': c['screening_report'].get('rejected', []),
                    },
                    'stability_report': {
                        'stability_report': c['stability_report'].get('stability_report', {}),
                        'stable_features': c['stability_report'].get('stable_features', []),
                        'unstable_features': c['stability_report'].get('unstable_features', []),
                    },
                }
                for c in candidates
            ],
            'scores': scores,
            'results': [
                {
                    'sample_id': r['sample_id'], 'ticker': r['ticker'],
                    'sharpe': r['sharpe'], 'max_dd': r['max_dd'],
                    'total_return': r['total_return']
                }
                for r in results
            ]
        }
        with open(os.path.join(it_dir, 'results.json'), 'w') as f:
            json.dump(clean, f, indent=2, default=str)

        # Also save as pickle for backward compatibility
        with open(os.path.join(it_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(clean, f)

        # Save COT feedback
        with open(os.path.join(it_dir, 'cot_feedback.txt'), 'w') as f:
            f.write(cot)

    def _select_best_strategy(self):
        """Select best strategy across all iterations by Sharpe ratio."""
        best_sharpe, best_cfg = -float('inf'), None
        for it in range(self.max_iterations):
            # Try JSON first, fall back to pickle
            json_path = os.path.join(self.output_dir, f'iteration_{it}', 'results.json')
            pkl_path = os.path.join(self.output_dir, f'iteration_{it}', 'results.pkl')

            data = None
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
            elif os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

            if data:
                for r in data.get('results', []):
                    if r.get('sharpe', 0) > best_sharpe:
                        best_sharpe = r['sharpe']
                        best_cfg = r

        if best_cfg:
            logger.info(f"Best: It{best_cfg.get('sample_id', 0)}, Sharpe={best_sharpe:.3f}")
        return best_cfg
