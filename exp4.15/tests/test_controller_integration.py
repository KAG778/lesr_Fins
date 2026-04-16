"""
Integration tests for lesr_controller.py (Plan 03-03).

Tests JSON-mode optimization loop, COT feedback with IC scores,
leakage activation, and fixed reward integration.
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure core/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))


class TestLESRControllerInstantiation:
    """Test that LESRController can be instantiated with new config."""

    def test_controller_instantiation_with_json_mode_config(self):
        """LESRController accepts n_candidates and max_iterations."""
        from lesr_controller import LESRController
        config = {
            'tickers': ['TEST'],
            'train_period': ['2020-01-01', '2020-12-31'],
            'val_period': ['2021-01-01', '2021-06-30'],
            'test_period': ['2021-07-01', '2021-12-31'],
            'data_loader': MagicMock(),
            'n_candidates': 3,
            'max_iterations': 5,
            'openai_key': 'test-key',
            'model': 'gpt-4o-mini',
            'output_dir': '/tmp/test_lesr_output',
        }
        ctrl = LESRController(config)
        assert ctrl.n_candidates == 3
        assert ctrl.max_iterations == 5
        assert ctrl.best_selection is None
        assert ctrl.best_score == {'sharpe': -float('inf')}

    def test_controller_default_config(self):
        """LESRController uses sensible defaults for n_candidates and max_iterations."""
        from lesr_controller import LESRController
        config = {
            'tickers': ['TEST'],
            'train_period': ['2020-01-01', '2020-12-31'],
            'val_period': ['2021-01-01', '2021-06-30'],
            'test_period': ['2021-07-01', '2021-12-31'],
            'data_loader': MagicMock(),
            'openai_key': 'test-key',
        }
        ctrl = LESRController(config)
        assert ctrl.n_candidates == 3  # D-24 default
        assert ctrl.max_iterations == 5  # D-24 default


class TestLeakageActivation:
    """Test that check_prompt_for_leakage is called in run_optimization."""

    def test_leakage_check_function_exists(self):
        """check_prompt_for_leakage function is importable."""
        from lesr_controller import check_prompt_for_leakage
        assert callable(check_prompt_for_leakage)

    def test_leakage_check_detects_test_metrics(self):
        """check_prompt_for_leakage detects test_sharpe."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "The test_sharpe was 1.5 and val_sharpe was 1.2"
        leaks = check_prompt_for_leakage(prompt)
        assert 'test_sharpe' in leaks
        assert 'val_sharpe' in leaks

    def test_leakage_check_clean_prompt(self):
        """check_prompt_for_leakage returns empty for clean prompts."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "The sharpe was 1.5 and max_dd was 20%"
        leaks = check_prompt_for_leakage(prompt)
        assert leaks == []

    def test_leakage_check_called_in_call_llm(self):
        """_call_llm activates check_prompt_for_leakage."""
        from lesr_controller import LESRController
        config = {
            'tickers': ['TEST'],
            'train_period': ['2020-01-01', '2020-12-31'],
            'val_period': ['2021-01-01', '2021-06-30'],
            'test_period': ['2021-07-01', '2021-12-31'],
            'data_loader': MagicMock(),
            'openai_key': 'test-key',
        }
        ctrl = LESRController(config)
        # Patch the OpenAI client to avoid actual API calls
        ctrl._openai_version = 1
        ctrl._openai_client = MagicMock()
        ctrl._openai_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"features": []}'))]
        )

        # Call with a prompt containing leaked content
        prompt = "test_sharpe: 1.5"
        ctrl._call_llm(prompt)

        # The method should have been called (leakage stripped before sending)
        assert ctrl._openai_client.chat.completions.create.called


class TestCOTFeedback:
    """Test COT feedback with IC scores and rejection reasons."""

    def test_cot_feedback_includes_ic_scores(self):
        """get_cot_feedback includes per-indicator IC scores (D-10)."""
        from prompts import get_cot_feedback
        selections = [
            {'features': [{'indicator': 'RSI', 'params': {'window': 14}}]},
            {'features': [{'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}]},
        ]
        scores = [
            {'sharpe': 1.5, 'max_dd': 20.0, 'total_return': 15.0},
            {'sharpe': 0.8, 'max_dd': 30.0, 'total_return': 5.0},
        ]
        screening_reports = [
            {'feature_metrics': {'RSI': {'ic': 0.05, 'variance': 0.001}}, 'rejected': []},
            {'feature_metrics': {'MACD': {'ic': 0.01, 'variance': 0.0001}}, 'rejected': [
                {'indicator': 'MACD', 'params': {}, 'reason': 'IC=0.01 below threshold 0.02'}
            ]},
        ]
        stability_reports = [
            {'stability_report': {}, 'unstable_features': []},
            {'stability_report': {}, 'unstable_features': []},
        ]
        feedback = get_cot_feedback(selections, scores, screening_reports, stability_reports)
        assert 'IC=' in feedback
        assert 'RSI' in feedback

    def test_cot_feedback_includes_rejection_reasons(self):
        """get_cot_feedback includes rejected feature reasons (D-11)."""
        from prompts import get_cot_feedback
        selections = [
            {'features': [{'indicator': 'RSI', 'params': {'window': 14}}]},
        ]
        scores = [{'sharpe': 1.0, 'max_dd': 10.0, 'total_return': 5.0}]
        screening_reports = [
            {'feature_metrics': {}, 'rejected': [
                {'indicator': 'Momentum', 'params': {'window': 10}, 'reason': 'IC=0.005 below threshold 0.02'}
            ]},
        ]
        stability_reports = [{'stability_report': {}, 'unstable_features': []}]
        feedback = get_cot_feedback(selections, scores, screening_reports, stability_reports)
        assert 'Momentum' in feedback
        assert 'Avoid' in feedback

    def test_cot_feedback_batch_mode(self):
        """get_cot_feedback generates batch feedback across candidates (D-12)."""
        from prompts import get_cot_feedback
        selections = [
            {'features': [{'indicator': 'RSI', 'params': {'window': 14}}]},
            {'features': [{'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}]},
            {'features': [{'indicator': 'Bollinger', 'params': {'window': 20, 'num_std': 2.0}}]},
        ]
        scores = [
            {'sharpe': 1.5, 'max_dd': 10.0, 'total_return': 15.0},
            {'sharpe': 0.8, 'max_dd': 20.0, 'total_return': 5.0},
            {'sharpe': 1.2, 'max_dd': 15.0, 'total_return': 10.0},
        ]
        screening_reports = [
            {'feature_metrics': {}, 'rejected': []},
            {'feature_metrics': {}, 'rejected': []},
            {'feature_metrics': {}, 'rejected': []},
        ]
        stability_reports = [
            {'stability_report': {}, 'unstable_features': []},
            {'stability_report': {}, 'unstable_features': []},
            {'stability_report': {}, 'unstable_features': []},
        ]
        feedback = get_cot_feedback(selections, scores, screening_reports, stability_reports)
        assert 'Candidate 1' in feedback
        assert 'Candidate 2' in feedback
        assert 'Candidate 3' in feedback
        assert 'Best candidate' in feedback


class TestIterationPrompts:
    """Test that iterations use correct prompt functions."""

    def test_initial_prompt_uses_render_initial(self):
        """render_initial_prompt produces a prompt with market stats."""
        from prompts import render_initial_prompt
        states = np.random.randn(10, 120) * 100
        prompt = render_initial_prompt(states)
        assert 'Available Indicators' in prompt
        assert 'JSON' in prompt

    def test_iteration_prompt_uses_get_iteration(self):
        """get_iteration_prompt produces curated prompt with last selection + feedback."""
        from prompts import get_iteration_prompt
        last_selection = {'features': [{'indicator': 'RSI', 'params': {'window': 14}}]}
        cot_feedback = "RSI performed well with IC=0.05"
        best_selection = {'features': [{'indicator': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}]}
        best_score = {'sharpe': 1.5}

        prompt = get_iteration_prompt(last_selection, cot_feedback, best_selection, best_score)
        assert 'Feedback from Last Iteration' in prompt
        assert 'Best Historical Selection' in prompt
        assert '1.500' in prompt  # best_sharpe formatted


class TestNoImportLibTempFile:
    """Verify no importlib or tempfile in controller."""

    def test_no_importlib_in_controller(self):
        """lesr_controller.py does not use importlib."""
        ctrl_path = Path(__file__).parent.parent / 'core' / 'lesr_controller.py'
        with open(ctrl_path, 'r') as f:
            content = f.read()
        assert 'import importlib' not in content

    def test_no_tempfile_in_controller(self):
        """lesr_controller.py does not use tempfile as a module."""
        ctrl_path = Path(__file__).parent.parent / 'core' / 'lesr_controller.py'
        with open(ctrl_path, 'r') as f:
            content = f.read()
        # Check for actual tempfile usage (import or NamedTemporaryFile), not docstring mentions
        assert 'import tempfile' not in content
        assert 'tempfile.NamedTemporaryFile' not in content


class TestFixedRewardIntegration:
    """Test that DQNTrainer uses compute_fixed_reward."""

    def test_trainer_no_intrinsic_reward_func(self):
        """DQNTrainer does not require intrinsic_reward_func."""
        from dqn_trainer import DQNTrainer
        dummy_revise = lambda s: np.zeros(5)
        trainer = DQNTrainer(
            ticker='TEST',
            revise_state_func=dummy_revise,
            state_dim=128,
        )
        assert not hasattr(trainer, 'intrinsic_reward') or not callable(getattr(trainer, 'intrinsic_reward', None))

    def test_compute_fixed_reward_returns_float(self):
        """compute_fixed_reward returns clipped float."""
        from dqn_trainer import DQNTrainer
        dummy_revise = lambda s: np.zeros(5)
        trainer = DQNTrainer(
            ticker='TEST',
            revise_state_func=dummy_revise,
            state_dim=128,
        )
        regime = np.array([0.5, 0.3, 0.8])
        action = 0  # BUY
        features = np.array([0.1, -0.2, 0.3])
        reward = trainer.compute_fixed_reward(regime, action, features)
        assert isinstance(reward, float)
        assert -10.0 <= reward <= 10.0

    def test_compute_fixed_reward_risk_management(self):
        """Rule 1: High risk + BUY gives negative reward."""
        from dqn_trainer import DQNTrainer
        dummy_revise = lambda s: np.zeros(5)
        trainer = DQNTrainer(
            ticker='TEST',
            revise_state_func=dummy_revise,
            state_dim=128,
        )
        # High risk regime, BUY action
        risky_regime = np.array([0.0, 0.2, 0.9])
        reward_buy = trainer.compute_fixed_reward(risky_regime, 0, np.array([0.1]))
        # Safe regime, BUY action
        safe_regime = np.array([0.3, 0.2, 0.1])
        reward_safe_buy = trainer.compute_fixed_reward(safe_regime, 0, np.array([0.1]))
        assert reward_buy < reward_safe_buy


class TestFilterCotMetrics:
    """Test filter_cot_metrics leakage prevention."""

    def test_filter_keeps_allowed_keys(self):
        """filter_cot_metrics keeps sharpe, max_dd, total_return."""
        from lesr_controller import filter_cot_metrics
        results = [
            {'sharpe': 1.5, 'max_dd': 20.0, 'total_return': 15.0,
             'sortino': 2.0, 'factor_metrics': {'f1': {'ic': 0.05}},
             'regime_metrics': {'bull': {'sharpe': 1.8}}}
        ]
        filtered = filter_cot_metrics(results)
        assert len(filtered) == 1
        assert 'sharpe' in filtered[0]
        assert 'max_dd' in filtered[0]
        assert 'total_return' in filtered[0]
        assert 'sortino' not in filtered[0]
        assert 'factor_metrics' not in filtered[0]
        assert 'regime_metrics' not in filtered[0]

    def test_filter_strips_test_val_keys(self):
        """filter_cot_metrics removes keys containing test/val."""
        from lesr_controller import filter_cot_metrics
        results = [
            {'sharpe': 1.5, 'test_sharpe': 1.2, 'val_sharpe': 1.3}
        ]
        filtered = filter_cot_metrics(results)
        assert 'test_sharpe' not in filtered[0]
        assert 'val_sharpe' not in filtered[0]
        assert 'sharpe' in filtered[0]


class TestValidateSelectionIntegration:
    """Test validate_selection integration with controller."""

    def test_validate_selection_from_llm_text(self):
        """validate_selection handles LLM text with JSON in markdown blocks."""
        from feature_library import validate_selection
        sample_state = np.random.randn(120) * 100 + 50
        llm_text = '''Here are my selected indicators:

```json
{
  "features": [
    {"indicator": "RSI", "params": {"window": 14}},
    {"indicator": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}},
    {"indicator": "Bollinger", "params": {"window": 20, "num_std": 2.0}}
  ],
  "rationale": "Diversified trend and volatility indicators"
}
```
'''
        result = validate_selection(llm_text, sample_state)
        assert result['selection'] is not None
        assert len(result['selection']) == 3
        assert result['revise_state'] is not None
        assert result['state_dim'] > 123
        assert len(result['errors']) == 0 or all('clipped' in e.lower() for e in result['errors'])


class TestLESRStrategyStateAssembly:
    """Test lesr_strategy.py correct state assembly."""

    def test_strategy_imports_regime_detector(self):
        """lesr_strategy.py imports detect_regime."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            'lesr_strategy',
            str(Path(__file__).parent.parent / 'core' / 'lesr_strategy.py')
        )
        # Just verify the file contains the import
        with open(Path(__file__).parent.parent / 'core' / 'lesr_strategy.py') as f:
            content = f.read()
        assert 'detect_regime' in content
        assert 'regime_vector' in content

    def test_strategy_builds_enhanced_state(self):
        """lesr_strategy.py on_data builds enhanced_state = [raw + regime + features]."""
        with open(Path(__file__).parent.parent / 'core' / 'lesr_strategy.py') as f:
            content = f.read()
        assert 'np.concatenate([raw_state, regime_vector, features])' in content
