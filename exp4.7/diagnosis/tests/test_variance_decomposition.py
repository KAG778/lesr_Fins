"""Tests for VarianceDecomposer module."""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add diagnosis directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from variance_decomposition import VarianceDecomposer


class TestVarianceDecomposer:
    """Test suite for VarianceDecomposer variance decomposition."""

    def test_decompose_partitions_variance(self):
        """Between + within should approximately equal total variance."""
        decomposer = VarianceDecomposer()

        # Create DataFrame with 3 groups of 10, different means
        np.random.seed(42)
        data = []
        for group_id in range(3):
            mean_sharpe = 1.0 + group_id * 0.5  # 1.0, 1.5, 2.0
            for _ in range(10):
                data.append({
                    'llm_sample_id': f'sample_{group_id}',
                    'sharpe': mean_sharpe + np.random.randn() * 0.1,
                })

        df = pd.DataFrame(data)
        result = decomposer.decompose(df, group_col='llm_sample_id')

        assert result['between_fraction'] > 0
        assert result['within_group_variance'] > 0

        # Between + within should approximately equal total
        total_reconstructed = (
            result['between_group_variance']
            + result['within_group_variance']
        )
        np.testing.assert_allclose(
            total_reconstructed,
            result['total_variance'],
            rtol=1e-10,
        )

    def test_decompose_warns_on_few_runs(self):
        """Should warn when fewer than 10 runs are provided."""
        decomposer = VarianceDecomposer()

        # DataFrame with only 5 rows
        data = [
            {'llm_sample_id': 's1', 'sharpe': 1.0},
            {'llm_sample_id': 's1', 'sharpe': 1.2},
            {'llm_sample_id': 's2', 'sharpe': 0.8},
            {'llm_sample_id': 's2', 'sharpe': 1.1},
            {'llm_sample_id': 's3', 'sharpe': 0.9},
        ]
        df = pd.DataFrame(data)
        result = decomposer.decompose(df, group_col='llm_sample_id')

        assert result['warning'] is not None
        assert '10' in result['warning']

    def test_full_decomposition_three_factors(self):
        """Three-factor decomposition with LLM samples having different means."""
        decomposer = VarianceDecomposer()

        # 3 LLM samples x 3 DQN seeds x 2 tickers = 18 rows
        np.random.seed(42)
        data = []
        for llm_id in range(3):
            mean_sharpe = 1.0 + llm_id * 0.8  # Distinct LLM means
            for dqn_seed in range(3):
                for ticker in ['AAPL', 'MSFT']:
                    data.append({
                        'run_id': f'run_{llm_id}_{dqn_seed}_{ticker}',
                        'llm_sample_id': f'llm_{llm_id}',
                        'dqn_seed': f'seed_{dqn_seed}',
                        'ticker': ticker,
                        'sharpe': mean_sharpe + np.random.randn() * 0.1,
                    })

        df = pd.DataFrame(data)
        result = decomposer.full_decomposition(df)

        assert result['llm_variance_fraction'] > 0.1
        assert result['total_variance'] > 0
        assert 'llm_decomposition' in result
        assert 'ticker_decomposition' in result
        assert 'anova_results' in result

    def test_fractions_sum_approximately_one(self):
        """Variance fractions should approximately sum to 1.0."""
        decomposer = VarianceDecomposer()

        np.random.seed(42)
        data = []
        for llm_id in range(3):
            mean_sharpe = 1.0 + llm_id * 0.8
            for dqn_seed in range(3):
                for ticker in ['AAPL', 'MSFT']:
                    data.append({
                        'run_id': f'run_{llm_id}_{dqn_seed}_{ticker}',
                        'llm_sample_id': f'llm_{llm_id}',
                        'dqn_seed': f'seed_{dqn_seed}',
                        'ticker': ticker,
                        'sharpe': mean_sharpe + np.random.randn() * 0.1,
                    })

        df = pd.DataFrame(data)
        result = decomposer.full_decomposition(df)

        total_fraction = (
            result['llm_variance_fraction']
            + result['dqn_variance_fraction']
            + result['data_variance_fraction']
            + result['residual_fraction']
        )
        np.testing.assert_allclose(total_fraction, 1.0, atol=0.1)

    def test_generate_report_contains_variance_breakdown(self):
        """Report should contain key variance decomposition terms."""
        decomposer = VarianceDecomposer()

        np.random.seed(42)
        data = []
        for llm_id in range(3):
            mean_sharpe = 1.0 + llm_id * 0.5
            for dqn_seed in range(3):
                for ticker in ['AAPL', 'MSFT']:
                    data.append({
                        'run_id': f'run_{llm_id}_{dqn_seed}_{ticker}',
                        'llm_sample_id': f'llm_{llm_id}',
                        'dqn_seed': f'seed_{dqn_seed}',
                        'ticker': ticker,
                        'sharpe': mean_sharpe + np.random.randn() * 0.1,
                    })

        df = pd.DataFrame(data)
        report = decomposer.generate_report(df)

        assert 'LLM' in report
        assert 'DQN' in report
        assert 'variance' in report
        assert 'fraction' in report

    def test_handles_single_group(self):
        """Should not crash when all rows have same group id."""
        decomposer = VarianceDecomposer()

        data = []
        for i in range(15):
            data.append({
                'llm_sample_id': 'same_group',
                'sharpe': 1.0 + np.random.randn() * 0.1,
            })

        df = pd.DataFrame(data)
        result = decomposer.decompose(df, group_col='llm_sample_id')

        # With single group, between_fraction should be ~0
        assert result['between_fraction'] == pytest.approx(0.0, abs=1e-10)
        assert result['n_groups'] == 1
