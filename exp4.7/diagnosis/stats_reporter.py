"""
Stats Reporter Module for LESR Diagnosis Framework

Statistical comparison between LESR and DQN baseline using:
- Welch's t-test
- Bootstrap BCa confidence interval
- Mann-Whitney U test
"""

import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class StatsReporter:
    """Statistical comparison between LESR and DQN baseline."""

    def compare_sharpe(self, lesr_sharpes: list, baseline_sharpes: list,
                       confidence: float = 0.95) -> dict:
        """Compare LESR vs baseline Sharpe ratios using multiple statistical tests.

        Args:
            lesr_sharpes: List of Sharpe ratios from LESR runs
            baseline_sharpes: List of Sharpe ratios from DQN baseline runs
            confidence: Confidence level for bootstrap CI (default 0.95)

        Returns:
            dict with keys:
                ci_low, ci_high: Bootstrap BCa 95% CI for Sharpe difference
                t_stat, ttest_p: Welch's t-test statistic and p-value
                mann_whitney_u, mann_whitney_p: Mann-Whitney U test
                lesr_mean, baseline_mean: Mean Sharpe for each group
                effect_size: Mean difference (lesr_mean - baseline_mean)
                n_lesr, n_baseline: Sample sizes
                significant_005: bool, True if ttest_p < 0.05
        """
        lesr = np.array(lesr_sharpes, dtype=float)
        baseline = np.array(baseline_sharpes, dtype=float)

        lesr_mean = float(np.mean(lesr))
        baseline_mean = float(np.mean(baseline))
        effect_size = lesr_mean - baseline_mean

        # Welch's t-test
        t_result = stats.ttest_ind(lesr, baseline, equal_var=False)
        t_stat = float(t_result.statistic)
        ttest_p = float(t_result.pvalue)

        # Mann-Whitney U test
        mw_result = stats.mannwhitneyu(lesr, baseline, alternative='greater')
        mann_whitney_u = float(mw_result.statistic)
        mann_whitney_p = float(mw_result.pvalue)

        # Bootstrap BCa confidence interval
        def diff_statistic(x, y, axis=-1):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)

        bootstrap_result = stats.bootstrap(
            (lesr, baseline),
            diff_statistic,
            n_resamples=10000,
            confidence_level=confidence,
            method='BCa'
        )
        ci_low = float(bootstrap_result.confidence_interval.low)
        ci_high = float(bootstrap_result.confidence_interval.high)

        result = {
            'ci_low': ci_low,
            'ci_high': ci_high,
            't_stat': t_stat,
            'ttest_p': ttest_p,
            'mann_whitney_u': mann_whitney_u,
            'mann_whitney_p': mann_whitney_p,
            'lesr_mean': lesr_mean,
            'baseline_mean': baseline_mean,
            'effect_size': effect_size,
            'n_lesr': len(lesr),
            'n_baseline': len(baseline),
            'significant_005': ttest_p < 0.05,
        }

        logger.info(
            "LESR vs Baseline: t=%.3f p=%.4f effect=%.3f CI=[%.3f, %.3f]",
            t_stat, ttest_p, effect_size, ci_low, ci_high
        )

        return result

    def generate_report(self, lesr_sharpes: list, baseline_sharpes: list,
                        output_format: str = 'markdown') -> str:
        """Generate a human-readable comparison report.

        Args:
            lesr_sharpes: List of Sharpe ratios from LESR runs
            baseline_sharpes: List of Sharpe ratios from baseline runs
            output_format: 'markdown' or 'text'

        Returns:
            Formatted report string.
        """
        comparison = self.compare_sharpe(lesr_sharpes, baseline_sharpes)

        if output_format == 'markdown':
            lines = [
                "# LESR vs DQN Baseline Statistical Comparison",
                "",
                f"**Sample sizes:** LESR n={comparison['n_lesr']}, "
                f"Baseline n={comparison['n_baseline']}",
                "",
                f"**Mean Sharpe Ratio:**",
                f"- LESR: {comparison['lesr_mean']:.4f}",
                f"- Baseline: {comparison['baseline_mean']:.4f}",
                f"- Effect size (difference): {comparison['effect_size']:.4f}",
                "",
                f"**Welch's t-test:**",
                f"- t-statistic: {comparison['t_stat']:.4f}",
                f"- p-value: {comparison['ttest_p']:.4f}",
                "",
                f"**Bootstrap 95% CI for difference (BCa):**",
                f"- [{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}]",
                "",
                f"**Mann-Whitney U test (one-sided, LESR > Baseline):**",
                f"- U-statistic: {comparison['mann_whitney_u']:.4f}",
                f"- p-value: {comparison['mann_whitney_p']:.4f}",
                "",
            ]

            if comparison['significant_005'] and comparison['effect_size'] > 0:
                lines.append(
                    "**Conclusion:** LESR significantly outperforms baseline "
                    "(p < 0.05, Welch's t-test)."
                )
            else:
                lines.append(
                    "**Conclusion:** No significant difference detected "
                    "(p >= 0.05, Welch's t-test)."
                )

            return "\n".join(lines)
        else:
            # Plain text format
            lines = [
                "LESR vs DQN Baseline Statistical Comparison",
                "=" * 50,
                f"Sample sizes: LESR n={comparison['n_lesr']}, "
                f"Baseline n={comparison['n_baseline']}",
                f"Mean Sharpe - LESR: {comparison['lesr_mean']:.4f}, "
                f"Baseline: {comparison['baseline_mean']:.4f}",
                f"Effect size: {comparison['effect_size']:.4f}",
                f"Welch's t-test: t={comparison['t_stat']:.4f}, "
                f"p={comparison['ttest_p']:.4f}",
                f"Bootstrap 95% CI: [{comparison['ci_low']:.4f}, "
                f"{comparison['ci_high']:.4f}]",
                f"Mann-Whitney U: U={comparison['mann_whitney_u']:.4f}, "
                f"p={comparison['mann_whitney_p']:.4f}",
            ]
            return "\n".join(lines)

    def compare_per_ticker(self, results_df, ticker_col='ticker',
                           sharpe_col='sharpe', method_col='method') -> dict:
        """Per-ticker statistical comparison.

        Args:
            results_df: pandas DataFrame with columns for ticker, sharpe, method
                        method values: 'lesr' and 'baseline'
            ticker_col, sharpe_col, method_col: column names

        Returns:
            dict keyed by ticker, each value is output of compare_sharpe()
        """
        results = {}
        tickers = results_df[ticker_col].unique()

        for ticker in tickers:
            ticker_df = results_df[results_df[ticker_col] == ticker]

            lesr_sharpes = ticker_df[ticker_df[method_col] == 'lesr'][sharpe_col].tolist()
            baseline_sharpes = ticker_df[ticker_df[method_col] == 'baseline'][sharpe_col].tolist()

            if len(lesr_sharpes) >= 2 and len(baseline_sharpes) >= 2:
                results[ticker] = self.compare_sharpe(lesr_sharpes, baseline_sharpes)
            else:
                logger.warning(
                    "Insufficient data for ticker %s: "
                    "LESR n=%d, Baseline n=%d",
                    ticker, len(lesr_sharpes), len(baseline_sharpes)
                )
                results[ticker] = None

        return results
