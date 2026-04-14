"""Tests for StatsReporter module."""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add diagnosis directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stats_reporter import StatsReporter


class TestStatsReporter:
    """Test suite for StatsReporter statistical comparison."""

    def test_compare_sharpe_significant_difference(self):
        """LESR sharpes clearly higher than baseline should be significant."""
        reporter = StatsReporter()
        lesr_sharpes = [1.5, 1.8, 2.0, 1.7, 1.9]
        baseline_sharpes = [0.5, 0.8, 0.6, 0.7, 0.9]

        result = reporter.compare_sharpe(lesr_sharpes, baseline_sharpes)

        assert result['significant_005'] is True
        assert result['effect_size'] > 0
        assert result['ttest_p'] < 0.05

    def test_compare_sharpe_no_difference(self):
        """Same distribution should show no significant difference."""
        reporter = StatsReporter()
        np.random.seed(42)
        values = np.random.randn(10).tolist()
        lesr_sharpes = values
        baseline_sharpes = values

        result = reporter.compare_sharpe(lesr_sharpes, baseline_sharpes)

        # Identical samples should not be significant
        assert result['significant_005'] is False or result['ttest_p'] >= 0.05

    def test_bootstrap_ci_does_not_contain_zero_when_significant(self):
        """Bootstrap CI should not contain zero when LESR clearly outperforms."""
        reporter = StatsReporter()
        lesr_sharpes = [1.5, 1.8, 2.0, 1.7, 1.9]
        baseline_sharpes = [0.5, 0.8, 0.6, 0.7, 0.9]

        result = reporter.compare_sharpe(lesr_sharpes, baseline_sharpes)

        # CI should be entirely above zero
        assert result['ci_low'] > 0

    def test_generate_report_contains_all_sections(self):
        """Report should contain all key statistical test names."""
        reporter = StatsReporter()
        lesr_sharpes = [1.5, 1.8, 2.0, 1.7, 1.9]
        baseline_sharpes = [0.5, 0.8, 0.6, 0.7, 0.9]

        report = reporter.generate_report(lesr_sharpes, baseline_sharpes)

        assert 'Welch' in report
        assert 'Bootstrap' in report
        assert 'Mann-Whitney' in report
        assert 'Sharpe' in report

    def test_compare_per_ticker(self):
        """Per-ticker comparison should return dict keyed by ticker."""
        reporter = StatsReporter()

        # Create test DataFrame with 2 tickers, 5 runs each
        data = []
        for ticker in ['AAPL', 'MSFT']:
            for i in range(5):
                data.append({
                    'ticker': ticker,
                    'sharpe': 1.0 + i * 0.1 + (0.5 if ticker == 'AAPL' else 0),
                    'method': 'lesr',
                })
                data.append({
                    'ticker': ticker,
                    'sharpe': 0.5 + i * 0.05,
                    'method': 'baseline',
                })

        df = pd.DataFrame(data)
        results = reporter.compare_per_ticker(df)

        assert len(results) == 2
        assert 'AAPL' in results
        assert 'MSFT' in results
        assert results['AAPL'] is not None
        assert 'ttest_p' in results['AAPL']
