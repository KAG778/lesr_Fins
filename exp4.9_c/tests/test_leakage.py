"""
Tests for COT feedback data leakage prevention (D-06).

Verifies that:
- filter_cot_metrics strips sensitive keys (factor_metrics, regime_metrics, sortino, etc.)
- filter_cot_metrics keeps training-allowed keys (sharpe, max_dd, total_return)
- check_prompt_for_leakage detects leaked metric names in rendered prompt text
- check_prompt_for_leakage returns empty list for clean prompts
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure exp4.9_c is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFilterCotMetrics:
    """Tests for filter_cot_metrics function."""

    def test_filter_strips_factor_metrics(self):
        """factor_metrics should be stripped from results."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sample_id': 0,
            'ticker': 'TSLA',
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
            'factor_metrics': {'feature_0': {'ic': 0.3}},
        }]
        filtered = filter_cot_metrics(results)
        assert 'factor_metrics' not in filtered[0]

    def test_filter_strips_regime_metrics(self):
        """regime_metrics should be stripped from results."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
            'regime_metrics': {'bull': {'sharpe': 2.0}},
        }]
        filtered = filter_cot_metrics(results)
        assert 'regime_metrics' not in filtered[0]

    def test_filter_strips_sortino(self):
        """sortino should be stripped from results."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
            'sortino': 2.1,
        }]
        filtered = filter_cot_metrics(results)
        assert 'sortino' not in filtered[0]

    def test_filter_strips_calmar(self):
        """calmar should be stripped from results."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
            'calmar': 0.8,
        }]
        filtered = filter_cot_metrics(results)
        assert 'calmar' not in filtered[0]

    def test_filter_strips_win_rate(self):
        """win_rate should be stripped from results."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
            'win_rate': 0.6,
        }]
        filtered = filter_cot_metrics(results)
        assert 'win_rate' not in filtered[0]

    def test_filter_strips_test_keys(self):
        """Any key containing 'test' should be stripped."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
            'test_sharpe': 2.0,
        }]
        filtered = filter_cot_metrics(results)
        assert 'test_sharpe' not in filtered[0]

    def test_filter_strips_val_keys(self):
        """Any key containing 'val' should be stripped."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
            'val_sharpe': 1.8,
        }]
        filtered = filter_cot_metrics(results)
        assert 'val_sharpe' not in filtered[0]

    def test_filter_keeps_sharpe(self):
        """sharpe should be kept."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
            'sortino': 2.1,
        }]
        filtered = filter_cot_metrics(results)
        assert filtered[0]['sharpe'] == 1.5

    def test_filter_keeps_max_dd(self):
        """max_dd should be kept."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
        }]
        filtered = filter_cot_metrics(results)
        assert filtered[0]['max_dd'] == 15.0

    def test_filter_keeps_total_return(self):
        """total_return should be kept."""
        from lesr_controller import filter_cot_metrics
        results = [{
            'sharpe': 1.5,
            'max_dd': 15.0,
            'total_return': 25.0,
        }]
        filtered = filter_cot_metrics(results)
        assert filtered[0]['total_return'] == 25.0

    def test_filter_multiple_results(self):
        """filter_cot_metrics should handle multiple results."""
        from lesr_controller import filter_cot_metrics
        results = [
            {'sharpe': 1.5, 'max_dd': 15.0, 'total_return': 25.0,
             'factor_metrics': {}, 'sortino': 2.0},
            {'sharpe': 0.8, 'max_dd': 20.0, 'total_return': -5.0,
             'regime_metrics': {}, 'calmar': 0.3},
        ]
        filtered = filter_cot_metrics(results)
        assert len(filtered) == 2
        assert 'factor_metrics' not in filtered[0]
        assert 'sortino' not in filtered[0]
        assert 'regime_metrics' not in filtered[1]
        assert 'calmar' not in filtered[1]
        assert filtered[0]['sharpe'] == 1.5
        assert filtered[1]['sharpe'] == 0.8


class TestCheckPromptForLeakage:
    """Tests for check_prompt_for_leakage function."""

    def test_detects_test_sharpe(self):
        """Prompt containing 'test_sharpe' should be detected as leaked."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "The strategy achieved test_sharpe of 2.5"
        leaks = check_prompt_for_leakage(prompt)
        assert len(leaks) > 0
        assert 'test_sharpe' in leaks

    def test_detects_val_sharpe(self):
        """Prompt containing 'val_sharpe' should be detected as leaked."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "Validation Sharpe (val_sharpe) was 1.8"
        leaks = check_prompt_for_leakage(prompt)
        assert len(leaks) > 0
        assert 'val_sharpe' in leaks

    def test_detects_factor_metrics(self):
        """Prompt containing 'factor_metrics' should be detected as leaked."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "factor_metrics: feature_0 IC = 0.3"
        leaks = check_prompt_for_leakage(prompt)
        assert len(leaks) > 0
        assert 'factor_metrics' in leaks

    def test_detects_regime_metrics(self):
        """Prompt containing 'regime_metrics' should be detected as leaked."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "regime_metrics: bull sharpe = 2.1"
        leaks = check_prompt_for_leakage(prompt)
        assert len(leaks) > 0
        assert 'regime_metrics' in leaks

    def test_detects_sortino(self):
        """Prompt containing 'sortino' should be detected as leaked."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "Sortino ratio was 2.0"
        leaks = check_prompt_for_leakage(prompt)
        assert 'sortino' in leaks

    def test_detects_calmar(self):
        """Prompt containing 'calmar' should be detected as leaked."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "Calmar ratio: 0.8"
        leaks = check_prompt_for_leakage(prompt)
        assert 'calmar' in leaks

    def test_detects_win_rate(self):
        """Prompt containing 'win_rate' should be detected as leaked."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "Win rate: 0.65"
        leaks = check_prompt_for_leakage(prompt)
        assert 'win_rate' in leaks

    def test_detects_quantile_spread(self):
        """Prompt containing 'quantile_spread' should be detected as leaked."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "quantile_spread for feature_0 was 0.05"
        leaks = check_prompt_for_leakage(prompt)
        assert 'quantile_spread' in leaks

    def test_clean_prompt_returns_empty(self):
        """Prompt with only allowed metrics should return empty list."""
        from lesr_controller import check_prompt_for_leakage
        prompt = """
        Performance:
          Sharpe Ratio: 1.500
          Max Drawdown: 15.00%
          Total Return: 25.00%
        Please analyze and improve.
        """
        leaks = check_prompt_for_leakage(prompt)
        assert leaks == []

    def test_multiple_leaks_detected(self):
        """Multiple leaked metrics in one prompt should all be detected."""
        from lesr_controller import check_prompt_for_leakage
        prompt = "sortino=2.0, calmar=0.8, win_rate=0.6"
        leaks = check_prompt_for_leakage(prompt)
        assert 'sortino' in leaks
        assert 'calmar' in leaks
        assert 'win_rate' in leaks
        assert len(leaks) == 3
