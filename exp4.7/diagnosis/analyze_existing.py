"""
Post-hoc analysis of existing experiment results.

Can analyze results from:
1. New diagnosis experiment directories (with manifest.jsonl)
2. Old result directories (exp4.7/result_W*_test* format)

Usage:
    python exp4.7/diagnosis/analyze_existing.py \
        --results-dir exp4.7/result_W1_test2019 \
        --report-type statistical
"""
import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from stats_reporter import StatsReporter
from variance_decomposition import VarianceDecomposer
from feature_quality import compute_feature_quality
from structured_logger import StructuredLogger

logger = logging.getLogger(__name__)


def load_iteration_results(results_dir: str) -> list:
    """Load results from a single experiment directory.

    Handles both:
    - New format: run_XXX/iteration_N/results.pkl
    - Old format: iteration_N/results.pkl (existing exp4.7 results)

    Returns list of dicts with keys:
        run_id, iteration, sample_id, ticker, sharpe, max_dd, total_return,
        llm_code_path, config_path
    """
    results_path = Path(results_dir)
    all_results = []

    # Check if this is a new-format experiment directory (has run_XXX subdirs)
    run_dirs = sorted(results_path.glob('run_*'))
    if run_dirs and any(d.is_dir() for d in run_dirs):
        # New format: iterate over run directories
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            for iter_dir in sorted(run_dir.glob('iteration_*')):
                pkl_file = iter_dir / 'results.pkl'
                if pkl_file.exists():
                    try:
                        with open(pkl_file, 'rb') as f:
                            data = pickle.load(f)
                        iteration = int(iter_dir.name.split('_')[1])
                        for r in data.get('results', []):
                            r['run_id'] = run_id
                            r['iteration'] = iteration
                            all_results.append(r)
                    except Exception as e:
                        logger.warning(f"Failed to load {pkl_file}: {e}")
    else:
        # Old format: iteration directories at top level
        run_id = results_path.name  # Use directory name as run_id
        for iter_dir in sorted(results_path.glob('iteration_*')):
            pkl_file = iter_dir / 'results.pkl'
            if pkl_file.exists():
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    iteration = int(iter_dir.name.split('_')[1])
                    for r in data.get('results', []):
                        r['run_id'] = run_id
                        r['iteration'] = iteration
                        all_results.append(r)
                except Exception as e:
                    logger.warning(f"Failed to load {pkl_file}: {e}")

    return all_results


def analyze_experiment(results_dir: str, report_type: str = 'all') -> dict:
    """Analyze experiment results and generate reports.

    Args:
        results_dir: Path to experiment directory
        report_type: 'statistical', 'variance', 'feature', or 'all'

    Returns:
        dict with keys:
            report: str (formatted markdown report)
            stats: dict (from StatsReporter, if statistical report)
            decomposition: dict (from VarianceDecomposer, if variance report)
            feature_quality: dict (if feature report)
    """
    results_path = Path(results_dir)

    # Load results
    results = load_iteration_results(results_dir)
    if not results:
        logger.warning(f"No results found in {results_dir}")
        return {'report': f"No results found in {results_dir}"}

    df = pd.DataFrame(results)

    report_sections = []
    output = {'report': ''}

    # Statistical comparison
    if report_type in ('statistical', 'all'):
        reporter = StatsReporter()
        # Group by method if available, otherwise assume all LESR
        if 'method' in df.columns:
            # Compare LESR vs baseline
            lesr_sharpes = df[df['method'] == 'lesr']['sharpe'].tolist()
            baseline_sharpes = df[df['method'] == 'baseline']['sharpe'].tolist()
            if lesr_sharpes and baseline_sharpes:
                report_sections.append(reporter.generate_report(
                    lesr_sharpes, baseline_sharpes))
                output['stats'] = reporter.compare_sharpe(
                    lesr_sharpes, baseline_sharpes)

        # Per-ticker analysis
        report_sections.append("\n## Per-Ticker Results\n")
        if 'ticker' in df.columns:
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker]
                mean_sharpe = ticker_df['sharpe'].mean()
                std_sharpe = ticker_df['sharpe'].std()
                report_sections.append(
                    f"- **{ticker}**: Sharpe = {mean_sharpe:.3f} "
                    f"+/- {std_sharpe:.3f} (n={len(ticker_df)})")
        else:
            report_sections.append("- No ticker column found in results")

    # Variance decomposition
    if report_type in ('variance', 'all'):
        decomposer = VarianceDecomposer()
        # Use iteration as LLM sample proxy if llm_sample_id not available
        if 'llm_sample_id' not in df.columns:
            df['llm_sample_id'] = df['iteration'].astype(str)
        # Need ticker and dqn_seed columns for full decomposition
        if 'ticker' not in df.columns:
            df['ticker'] = 'unknown'
        if 'dqn_seed' not in df.columns:
            df['dqn_seed'] = 0
        try:
            report_sections.append(decomposer.generate_report(df))
            output['decomposition'] = decomposer.decompose(df)
        except Exception as e:
            report_sections.append(
                f"\n## Variance Decomposition\nError: {e}")

    # Feature quality
    if report_type in ('feature', 'all'):
        # Try to load states from pickle for feature analysis
        feature_reports = []
        # Check for feature_quality.json in run dirs
        fq_files = list(results_path.rglob('feature_quality.json'))
        if fq_files:
            import json
            for fq_file in fq_files[:10]:  # Limit to first 10
                try:
                    with open(fq_file) as f:
                        fq = json.load(f)
                    run_id = fq_file.parent.name
                    agg = fq.get('aggregate', {})
                    feature_reports.append(
                        f"- **{run_id}**: degenerate={agg.get('num_degenerate', '?')}, "
                        f"significant={agg.get('num_significant', '?')}, "
                        f"mean_corr={agg.get('mean_abs_correlation', 0):.3f}")
                except Exception as e:
                    logger.warning(f"Failed to load {fq_file}: {e}")
        if feature_reports:
            report_sections.append(
                "\n## Feature Quality\n" + '\n'.join(feature_reports))
        else:
            report_sections.append(
                "\n## Feature Quality\n"
                "No feature_quality.json files found. "
                "Run with feature extraction enabled.")

    output['report'] = '\n\n'.join(report_sections)
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Analyze existing experiment results')
    parser.add_argument('--results-dir', required=True,
                        help='Path to experiment results directory')
    parser.add_argument('--report-type',
                        choices=['statistical', 'variance', 'feature', 'all'],
                        default='all')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    result = analyze_experiment(args.results_dir, report_type=args.report_type)
    print(result['report'])

    # Save report
    report_path = Path(args.results_dir) / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(result['report'])
    logger.info(f"Report saved to {report_path}")


if __name__ == '__main__':
    main()
