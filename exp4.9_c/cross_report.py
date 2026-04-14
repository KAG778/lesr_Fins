"""
Cross-experiment aggregation reporter.

Reads result_221_SW* directories, aggregates metrics across
stocks/windows/runs, and generates publication-ready markdown
comparison tables.

Usage:
    python exp4.9_c/cross_report.py --base_dir exp4.9_c/ --output report.md
"""
import argparse
import glob
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Metrics to extract from each test result dict
LESR_METRICS = ["sharpe", "sortino", "max_dd", "calmar", "total_return", "win_rate"]
BASE_METRICS = ["sharpe", "sortino", "max_dd", "calmar", "total_return", "win_rate"]


def _extract_window_name(dir_name: str) -> str:
    """Extract window name (e.g., 'SW01') from directory name."""
    match = re.search(r"SW(\d+)", dir_name)
    if match:
        return f"SW{match.group(1)}"
    return dir_name


def _extract_factor_ic_mean(factor_metrics: dict) -> Optional[float]:
    """Compute mean IC across feature dimensions from factor_metrics dict."""
    if not factor_metrics or not isinstance(factor_metrics, dict):
        return None
    ic_values = []
    for feat_name, feat_data in factor_metrics.items():
        if isinstance(feat_data, dict) and "ic" in feat_data:
            ic_values.append(feat_data["ic"])
    if ic_values:
        return float(np.mean(ic_values))
    return None


def aggregate_results(base_dir: str, pattern: str = "result_221_SW*") -> dict:
    """Aggregate results from multiple sliding-window experiment directories.

    Args:
        base_dir: Base directory containing result_221_SW* subdirectories.
        pattern: Glob pattern to match result directories.

    Returns:
        Nested dict: {window_name: {ticker: {metric_name: [values]}}}
        Metrics prefixed with 'lesr_' or 'base_' to distinguish method.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logger.warning(f"Base directory does not exist: {base_dir}")
        return {}

    result_dirs = sorted(base_path.glob(pattern))
    if not result_dirs:
        logger.info(f"No directories matching '{pattern}' in {base_dir}")
        return {}

    aggregated = {}

    for result_dir in result_dirs:
        if not result_dir.is_dir():
            continue

        window_name = _extract_window_name(result_dir.name)
        aggregated[window_name] = {}

        # Try loading test_set_results.pkl
        pkl_file = result_dir / "test_set_results.pkl"
        if not pkl_file.exists():
            logger.warning(f"No test_set_results.pkl in {result_dir}")
            continue

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {pkl_file}: {e}")
            continue

        if not isinstance(data, dict):
            logger.warning(f"Unexpected data format in {pkl_file}")
            continue

        for ticker, ticker_data in data.items():
            if not isinstance(ticker_data, dict):
                continue

            # Skip entries with errors
            if ticker_data.get("error") is not None:
                logger.info(f"Skipping {ticker} in {window_name}: error={ticker_data['error']}")
                continue

            metrics = {}

            # Extract LESR metrics
            lesr_test = ticker_data.get("lesr_test", {})
            if isinstance(lesr_test, dict):
                for metric in LESR_METRICS:
                    val = lesr_test.get(metric)
                    if val is not None:
                        metrics[f"lesr_{metric}"] = [float(val)]
                    else:
                        metrics[f"lesr_{metric}"] = [0.0]

                # Extract factor IC mean if factor_metrics present
                factor_metrics = lesr_test.get("factor_metrics")
                ic_mean = _extract_factor_ic_mean(factor_metrics)
                if ic_mean is not None:
                    metrics["lesr_factor_ic_mean"] = [ic_mean]

            # Extract Baseline metrics
            baseline_test = ticker_data.get("baseline_test", {})
            if isinstance(baseline_test, dict):
                for metric in BASE_METRICS:
                    val = baseline_test.get(metric)
                    if val is not None:
                        metrics[f"base_{metric}"] = [float(val)]
                    else:
                        metrics[f"base_{metric}"] = [0.0]

            aggregated[window_name][ticker] = metrics

    return aggregated


def _fmt_mean_std(values: list) -> str:
    """Format a list of values as mean +/- std."""
    if not values:
        return "N/A"
    arr = np.array(values)
    mean = np.mean(arr)
    if len(arr) > 1:
        std = np.std(arr)
        return f"{mean:.3f} +/- {std:.3f}"
    return f"{mean:.3f}"


def generate_report(agg_results: dict, output_path: Optional[str] = None) -> str:
    """Generate a markdown comparison table from aggregated results.

    Args:
        agg_results: Output of aggregate_results().
        output_path: If given, write markdown to this file path.

    Returns:
        Markdown string with comparison table.
    """
    if not agg_results:
        report = "# Cross-Experiment Report\n\nNo results found.\n"
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
        return report

    lines = [
        "# Cross-Experiment Comparison Report",
        "",
        "## LESR vs Baseline: Sliding Window Results",
        "",
    ]

    # --- LESR Results Table ---
    lines.append("### LESR Results")
    lines.append("")
    header = "| Window | Ticker | Sharpe | Sortino | MaxDD | Calmar | WinRate | TotalReturn | Mean IC |"
    sep =    "|--------|--------|--------|---------|-------|--------|---------|-------------|---------|"
    lines.append(header)
    lines.append(sep)

    # Collect all values for overall summary
    all_lesr_sharpes = []
    all_lesr_sortinos = []
    all_lesr_maxdds = []
    all_lesr_calmars = []
    all_lesr_winrates = []
    all_lesr_returns = []
    all_lesr_ics = []

    for window_name in sorted(agg_results.keys()):
        window_data = agg_results[window_name]
        for ticker in sorted(window_data.keys()):
            m = window_data[ticker]
            sharpe_str = _fmt_mean_std(m.get("lesr_sharpe", []))
            sortino_str = _fmt_mean_std(m.get("lesr_sortino", []))
            maxdd_str = _fmt_mean_std(m.get("lesr_max_dd", []))
            calmar_str = _fmt_mean_std(m.get("lesr_calmar", []))
            winrate_str = _fmt_mean_std(m.get("lesr_win_rate", []))
            return_str = _fmt_mean_std(m.get("lesr_total_return", []))
            ic_str = _fmt_mean_std(m.get("lesr_factor_ic_mean", []))

            lines.append(
                f"| {window_name} | {ticker} | {sharpe_str} | {sortino_str} "
                f"| {maxdd_str} | {calmar_str} | {winrate_str} | {return_str} "
                f"| {ic_str} |"
            )

            all_lesr_sharpes.extend(m.get("lesr_sharpe", []))
            all_lesr_sortinos.extend(m.get("lesr_sortino", []))
            all_lesr_maxdds.extend(m.get("lesr_max_dd", []))
            all_lesr_calmars.extend(m.get("lesr_calmar", []))
            all_lesr_winrates.extend(m.get("lesr_win_rate", []))
            all_lesr_returns.extend(m.get("lesr_total_return", []))
            all_lesr_ics.extend(m.get("lesr_factor_ic_mean", []))

    # Overall summary row
    lines.append(
        f"| **Overall** | **all** | {_fmt_mean_std(all_lesr_sharpes)} "
        f"| {_fmt_mean_std(all_lesr_sortinos)} | {_fmt_mean_std(all_lesr_maxdds)} "
        f"| {_fmt_mean_std(all_lesr_calmars)} | {_fmt_mean_std(all_lesr_winrates)} "
        f"| {_fmt_mean_std(all_lesr_returns)} | {_fmt_mean_std(all_lesr_ics)} |"
    )

    # --- Baseline Results Table ---
    lines.append("")
    lines.append("### Baseline Results")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    all_base_sharpes = []
    all_base_sortinos = []
    all_base_maxdds = []
    all_base_calmars = []
    all_base_winrates = []
    all_base_returns = []

    for window_name in sorted(agg_results.keys()):
        window_data = agg_results[window_name]
        for ticker in sorted(window_data.keys()):
            m = window_data[ticker]
            sharpe_str = _fmt_mean_std(m.get("base_sharpe", []))
            sortino_str = _fmt_mean_std(m.get("base_sortino", []))
            maxdd_str = _fmt_mean_std(m.get("base_max_dd", []))
            calmar_str = _fmt_mean_std(m.get("base_calmar", []))
            winrate_str = _fmt_mean_std(m.get("base_win_rate", []))
            return_str = _fmt_mean_std(m.get("base_total_return", []))

            lines.append(
                f"| {window_name} | {ticker} | {sharpe_str} | {sortino_str} "
                f"| {maxdd_str} | {calmar_str} | {winrate_str} | {return_str} "
                f"| N/A |"
            )

            all_base_sharpes.extend(m.get("base_sharpe", []))
            all_base_sortinos.extend(m.get("base_sortino", []))
            all_base_maxdds.extend(m.get("base_max_dd", []))
            all_base_calmars.extend(m.get("base_calmar", []))
            all_base_winrates.extend(m.get("base_win_rate", []))
            all_base_returns.extend(m.get("base_total_return", []))

    lines.append(
        f"| **Overall** | **all** | {_fmt_mean_std(all_base_sharpes)} "
        f"| {_fmt_mean_std(all_base_sortinos)} | {_fmt_mean_std(all_base_maxdds)} "
        f"| {_fmt_mean_std(all_base_calmars)} | {_fmt_mean_std(all_base_winrates)} "
        f"| {_fmt_mean_std(all_base_returns)} | N/A |"
    )

    # --- Comparison Summary ---
    lines.append("")
    lines.append("### Comparison Summary")
    lines.append("")
    if all_lesr_sharpes and all_base_sharpes:
        lesr_mean = np.mean(all_lesr_sharpes)
        base_mean = np.mean(all_base_sharpes)
        diff = lesr_mean - base_mean
        lines.append(f"- LESR mean Sharpe: {lesr_mean:.4f}")
        lines.append(f"- Baseline mean Sharpe: {base_mean:.4f}")
        lines.append(f"- Difference: {diff:+.4f}")
        if diff > 0:
            lines.append(f"- **LESR outperforms baseline by {diff:.4f} Sharpe**")
        else:
            lines.append(f"- **Baseline outperforms LESR by {-diff:.4f} Sharpe**")

    report = "\n".join(lines) + "\n"

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Cross-experiment aggregation reporter")
    parser.add_argument("--base_dir", required=True,
                        help="Base directory containing result_221_SW* directories")
    parser.add_argument("--output", default=None,
                        help="Output path for markdown report")
    parser.add_argument("--pattern", default="result_221_SW*",
                        help="Glob pattern for result directories")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    agg = aggregate_results(args.base_dir, pattern=args.pattern)
    report = generate_report(agg, output_path=args.output)
    print(report)


if __name__ == "__main__":
    main()
