"""
Variance Decomposition Module for LESR Diagnosis Framework

Decomposes Sharpe ratio variance into instability sources:
- LLM sampling variance
- DQN training variance
- Data (ticker) variance
- Residual / unexplained variance

Uses ANOVA-style analysis with method of moments.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class VarianceDecomposer:
    """Decompose Sharpe ratio variance into instability sources."""

    def decompose(self, results_df: pd.DataFrame,
                  group_col: str = 'llm_sample_id',
                  value_col: str = 'sharpe') -> dict:
        """Decompose variance into between-group and within-group components.

        Args:
            results_df: DataFrame with at least group_col and value_col
            group_col: Column identifying the variance factor (e.g., 'llm_sample_id')
            value_col: Column with the metric to analyze (e.g., 'sharpe')

        Returns:
            dict with keys:
                total_variance: Overall variance of value_col
                between_group_variance: Variance of group means
                within_group_variance: total - between_group
                between_fraction: between / total (0 to 1)
                anova_f: (F_statistic, p_value) from one-way ANOVA
                levene_test: (statistic, p_value) for equal variance test
                n_groups: Number of groups
                n_total: Total observations
                warning: str or None -- warns if n_total < 10
        """
        n_total = len(results_df)
        warning = None

        if n_total < 10:
            warning = (
                f"Only {n_total} runs. Recommend >= 10 for "
                "reliable decomposition."
            )
            logger.warning(warning)

        # Total variance
        total_variance = float(results_df[value_col].var())

        # Group means and between-group variance
        # Use ANOVA-style decomposition: total SS = between SS + within SS
        grand_mean = float(results_df[value_col].mean())
        groups_data = list(results_df.groupby(group_col)[value_col])

        # Between-group sum of squares (weighted by group size)
        between_ss = 0.0
        within_ss = 0.0
        for group_name, group_values in groups_data:
            n_j = len(group_values)
            group_mean = group_values.mean()
            between_ss += n_j * (group_mean - grand_mean) ** 2
            within_ss += float(((group_values - group_mean) ** 2).sum())

        n_groups = len(groups_data)

        # Convert to variance estimates
        # Between-group variance estimate
        if n_groups > 1:
            between_group_variance = between_ss / (n_total - 1)
        else:
            between_group_variance = 0.0

        # Within-group variance estimate
        within_group_variance = within_ss / (n_total - 1)

        # Between fraction
        if total_variance > 0:
            between_fraction = between_group_variance / total_variance
        else:
            between_fraction = 0.0

        # ANOVA and Levene tests
        groups_arrays = [gv.values for _, gv in groups_data]

        if len(groups_arrays) >= 2:
            anova_result = stats.f_oneway(*groups_arrays)
            anova_f = (float(anova_result.statistic), float(anova_result.pvalue))

            levene_result = stats.levene(*groups_arrays)
            levene_test = (float(levene_result.statistic), float(levene_result.pvalue))
        else:
            anova_f = (0.0, 1.0)
            levene_test = (0.0, 1.0)

        result = {
            'total_variance': total_variance,
            'between_group_variance': between_group_variance,
            'within_group_variance': within_group_variance,
            'between_fraction': between_fraction,
            'anova_f': anova_f,
            'levene_test': levene_test,
            'n_groups': n_groups,
            'n_total': n_total,
            'warning': warning,
        }

        logger.info(
            "Decomposition (%s): total_var=%.4f, between=%.4f (%.1f%%), "
            "within=%.4f, n_groups=%d, n_total=%d",
            group_col, total_variance, between_group_variance,
            between_fraction * 100, within_group_variance, n_groups, n_total
        )

        return result

    def full_decomposition(self, results_df: pd.DataFrame) -> dict:
        """Three-factor decomposition: LLM sampling, DQN training, data (ticker).

        Args:
            results_df: DataFrame with columns:
                run_id, llm_sample_id, dqn_seed, ticker, sharpe

        Returns:
            dict with keys:
                total_variance: float
                llm_variance_fraction: Between llm_sample_id variance / total
                dqn_variance_fraction: Between dqn_seed variance / total (within LLM)
                data_variance_fraction: Between ticker variance / total
                residual_fraction: 1 - sum of above
                llm_decomposition: Output of decompose(group_col='llm_sample_id')
                ticker_decomposition: Output of decompose(group_col='ticker')
                anova_results: dict with per-factor (F, p) tuples
                warning: str or None
        """
        total_variance = float(results_df['sharpe'].var())

        warning = None
        if len(results_df) < 10:
            warning = (
                f"Only {len(results_df)} runs. Recommend >= 10 for "
                "reliable decomposition."
            )
            logger.warning(warning)

        # LLM variance
        llm_decomp = self.decompose(results_df, group_col='llm_sample_id')
        llm_between_var = llm_decomp['between_group_variance']

        # Ticker variance
        ticker_decomp = self.decompose(results_df, group_col='ticker')
        ticker_between_var = ticker_decomp['between_group_variance']

        # DQN variance (within LLM groups)
        # For each LLM sample, compute the variance across DQN seeds,
        # then take the mean
        dqn_variances = []
        for llm_id, group in results_df.groupby('llm_sample_id'):
            if len(group) > 1:
                dqn_group_var = group.groupby('dqn_seed')['sharpe'].mean().var()
                dqn_variances.append(dqn_group_var)
        dqn_between_var = float(np.mean(dqn_variances)) if dqn_variances else 0.0

        # Fractions
        if total_variance > 0:
            llm_fraction = llm_between_var / total_variance
            dqn_fraction = dqn_between_var / total_variance
            data_fraction = ticker_between_var / total_variance
        else:
            llm_fraction = 0.0
            dqn_fraction = 0.0
            data_fraction = 0.0

        residual_fraction = 1.0 - (llm_fraction + dqn_fraction + data_fraction)

        # ANOVA results per factor
        anova_results = {}

        for factor_col in ['llm_sample_id', 'dqn_seed', 'ticker']:
            groups = [
                g['sharpe'].values
                for _, g in results_df.groupby(factor_col)
            ]
            if len(groups) >= 2:
                anova_result = stats.f_oneway(*groups)
                anova_results[factor_col] = (
                    float(anova_result.statistic),
                    float(anova_result.pvalue)
                )
            else:
                anova_results[factor_col] = (0.0, 1.0)

        result = {
            'total_variance': total_variance,
            'llm_variance_fraction': llm_fraction,
            'dqn_variance_fraction': dqn_fraction,
            'data_variance_fraction': data_fraction,
            'residual_fraction': residual_fraction,
            'llm_decomposition': llm_decomp,
            'ticker_decomposition': ticker_decomp,
            'anova_results': anova_results,
            'warning': warning,
        }

        logger.info(
            "Full decomposition: total=%.4f, LLM=%.1f%%, DQN=%.1f%%, "
            "data=%.1f%%, residual=%.1f%%",
            total_variance,
            llm_fraction * 100,
            dqn_fraction * 100,
            data_fraction * 100,
            residual_fraction * 100,
        )

        return result

    def generate_report(self, results_df: pd.DataFrame,
                        output_format: str = 'markdown') -> str:
        """Generate human-readable variance decomposition report.

        Args:
            results_df: DataFrame with columns:
                run_id, llm_sample_id, dqn_seed, ticker, sharpe
            output_format: 'markdown' or 'text'

        Returns:
            Formatted report string.
        """
        decomp = self.full_decomposition(results_df)

        if output_format == 'markdown':
            lines = [
                "# Variance Decomposition Report",
                "",
                f"**Total Sharpe Variance:** {decomp['total_variance']:.4f}",
                "",
                "## Variance Attributed to Factors",
                "",
                f"| Factor | variance fraction | Percentage |",
                f"|--------|----------|------------|",
                f"| LLM sampling | {decomp['llm_variance_fraction']:.4f} | "
                f"{decomp['llm_variance_fraction'] * 100:.1f}% |",
                f"| DQN training | {decomp['dqn_variance_fraction']:.4f} | "
                f"{decomp['dqn_variance_fraction'] * 100:.1f}% |",
                f"| Data (ticker) | {decomp['data_variance_fraction']:.4f} | "
                f"{decomp['data_variance_fraction'] * 100:.1f}% |",
                f"| Residual | {decomp['residual_fraction']:.4f} | "
                f"{decomp['residual_fraction'] * 100:.1f}% |",
                "",
                "## ANOVA Results",
                "",
            ]

            for factor, (f_stat, p_val) in decomp['anova_results'].items():
                sig = " *" if p_val < 0.05 else ""
                lines.append(
                    f"- **{factor}**: F={f_stat:.3f}, p={p_val:.4f}{sig}"
                )

            lines.append("")
            if any(
                p < 0.05
                for _, (_, p) in decomp['anova_results'].items()
            ):
                lines.append(
                    "* Significant at p < 0.05 level"
                )
            else:
                lines.append(
                    "No factor shows statistically significant variance "
                    "contribution."
                )

            if decomp['warning']:
                lines.append("")
                lines.append(f"**Warning:** {decomp['warning']}")

            return "\n".join(lines)
        else:
            lines = [
                "Variance Decomposition Report",
                "=" * 50,
                f"Total Sharpe Variance: {decomp['total_variance']:.4f}",
                f"LLM sampling: {decomp['llm_variance_fraction']:.4f} "
                f"({decomp['llm_variance_fraction'] * 100:.1f}%)",
                f"DQN training: {decomp['dqn_variance_fraction']:.4f} "
                f"({decomp['dqn_variance_fraction'] * 100:.1f}%)",
                f"Data (ticker): {decomp['data_variance_fraction']:.4f} "
                f"({decomp['data_variance_fraction'] * 100:.1f}%)",
                f"Residual: {decomp['residual_fraction']:.4f} "
                f"({decomp['residual_fraction'] * 100:.1f}%)",
            ]

            for factor, (f_stat, p_val) in decomp['anova_results'].items():
                lines.append(f"  {factor}: F={f_stat:.3f}, p={p_val:.4f}")

            if decomp['warning']:
                lines.append(f"Warning: {decomp['warning']}")

            return "\n".join(lines)
