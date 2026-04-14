#!/usr/bin/env python3
"""
汇总滑动窗口实验结果，生成报告。
Extended to include sortino, calmar, win_rate, and factor_metrics mean IC.

Usage: python exp4.9_c/sliding_summary.py
"""
import pickle
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path("/home/wangmeiyi/AuctionNet/lesr")
EXP_DIR = ROOT / "exp4.9_c"
STOCKS = ['TSLA', 'NFLX', 'AMZN', 'MSFT']


def extract_window_metrics(data: dict) -> dict:
    """Extract per-stock metrics from a test_set_results.pkl data dict.

    Supports both old format (sharpe/max_dd/total_return only) and new
    format (with sortino, calmar, win_rate, factor_metrics).

    Args:
        data: Dict loaded from test_set_results.pkl with stock keys.

    Returns:
        Dict: {ticker: {metric_name: float_value, ...}}
    """
    result = {}

    for ticker, ticker_data in data.items():
        if not isinstance(ticker_data, dict):
            continue
        if ticker_data.get("error") is not None:
            result[ticker] = None
            continue

        metrics = {}

        # LESR metrics
        lesr = ticker_data.get("lesr_test", {})
        if isinstance(lesr, dict):
            metrics["lesr_sharpe"] = float(lesr.get("sharpe", 0.0))
            metrics["lesr_max_dd"] = float(lesr.get("max_dd", 0.0))
            metrics["lesr_total_return"] = float(lesr.get("total_return", 0.0))
            metrics["lesr_sortino"] = float(lesr.get("sortino", 0.0))
            metrics["lesr_calmar"] = float(lesr.get("calmar", 0.0))
            metrics["lesr_win_rate"] = float(lesr.get("win_rate", 0.0))

            # Factor metrics mean IC
            factor_metrics = lesr.get("factor_metrics")
            if isinstance(factor_metrics, dict) and factor_metrics:
                ic_values = [
                    v["ic"] for v in factor_metrics.values()
                    if isinstance(v, dict) and "ic" in v
                ]
                metrics["lesr_factor_ic_mean"] = float(np.mean(ic_values)) if ic_values else 0.0
            else:
                metrics["lesr_factor_ic_mean"] = 0.0

        # Baseline metrics
        baseline = ticker_data.get("baseline_test", {})
        if isinstance(baseline, dict):
            metrics["base_sharpe"] = float(baseline.get("sharpe", 0.0))
            metrics["base_max_dd"] = float(baseline.get("max_dd", 0.0))
            metrics["base_total_return"] = float(baseline.get("total_return", 0.0))
            metrics["base_sortino"] = float(baseline.get("sortino", 0.0))
            metrics["base_calmar"] = float(baseline.get("calmar", 0.0))
            metrics["base_win_rate"] = float(baseline.get("win_rate", 0.0))

        result[ticker] = metrics

    return result


def generate_markdown_report(rows: list, stocks: list = None) -> str:
    """Generate markdown report from rows of extracted window metrics.

    Args:
        rows: List of dicts, each with 'window', 'test_year', 'train_range',
              'stocks' (dict of {ticker: metrics_dict or None}).
        stocks: List of stock tickers for column headers. Defaults to STOCKS.

    Returns:
        Markdown string with comparison tables including extended metrics.
    """
    if stocks is None:
        stocks = STOCKS

    md_lines = [
        f"# Exp4.9_c 滑动窗口实验报告（扩展指标）\n",
        f"> 生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"## 实验设计\n",
        f"- **滑动窗口**: 训练3年 | 验证1年 | 测试1年",
        f"- **滑动步长**: 1年",
        f"- **标的**: {', '.join(stocks)}\n",
    ]

    # --- LESR Extended Metrics Table ---
    md_lines.extend([
        f"## LESR 结果（扩展指标）\n",
        f"| 窗口 | 测试年 | 训练区间 |",
    ])
    for t in stocks:
        md_lines[-1] += f" {t}_Sharpe | {t}_Sortino | {t}_Calmar | {t}_WinRate | {t}_Return | {t}_MeanIC |"
    md_lines.append(f"|------|--------|----------|")
    sep = "|------|--------|----------|"
    for t in stocks:
        sep += "----------|----------|----------|----------|----------|----------|"
    md_lines[-1] = sep

    for r in rows:
        line = f"| {r['window']} | {r['test_year']} | {r['train_range']} |"
        for t in stocks:
            s = r['stocks'].get(t)
            if s is not None:
                line += (f" {s.get('lesr_sharpe', 0):.3f} |"
                         f" {s.get('lesr_sortino', 0):.3f} |"
                         f" {s.get('lesr_calmar', 0):.3f} |"
                         f" {s.get('lesr_win_rate', 0):.3f} |"
                         f" {s.get('lesr_total_return', 0):.2f}% |"
                         f" {s.get('lesr_factor_ic_mean', 0):.3f} |")
            else:
                line += " N/A | N/A | N/A | N/A | N/A | N/A |"
        md_lines.append(line)

    # --- Baseline Extended Metrics Table ---
    md_lines.extend([
        f"\n## Baseline 结果（扩展指标）\n",
        f"| 窗口 | 测试年 | 训练区间 |",
    ])
    for t in stocks:
        md_lines[-1] += f" {t}_Sharpe | {t}_Sortino | {t}_Calmar | {t}_WinRate | {t}_Return |"
    sep2 = "|------|--------|----------|"
    for t in stocks:
        sep2 += "----------|----------|----------|----------|----------|"
    md_lines.append(sep2)

    for r in rows:
        line = f"| {r['window']} | {r['test_year']} | {r['train_range']} |"
        for t in stocks:
            s = r['stocks'].get(t)
            if s is not None:
                line += (f" {s.get('base_sharpe', 0):.3f} |"
                         f" {s.get('base_sortino', 0):.3f} |"
                         f" {s.get('base_calmar', 0):.3f} |"
                         f" {s.get('base_win_rate', 0):.3f} |"
                         f" {s.get('base_total_return', 0):.2f}% |")
            else:
                line += " N/A | N/A | N/A | N/A | N/A |"
        md_lines.append(line)

    return '\n'.join(md_lines) + '\n'


def main():
    print(f"\n{'='*90}")
    print(f"  Exp4.9_c 滑动窗口实验结果汇总（扩展指标版）")
    print(f"  生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  设计: 训练3年 | 验证1年 | 测试1年, 每年滑动, 2015-2024")
    print(f"{'='*90}\n")

    all_rows = []
    total_lesr_wins = 0
    total_base_wins = 0
    total_ties = 0
    stock_wins = {t: 0 for t in STOCKS}
    stock_total = {t: 0 for t in STOCKS}

    for idx in range(1, 11):
        test_year = 2014 + idx
        train_start = test_year - 4
        train_end = test_year - 2
        result_dir = EXP_DIR / f"result_SW{idx:02d}_test{test_year}"
        pkl_file = result_dir / 'test_set_results.pkl'

        row = {
            'window': f'SW{idx:02d}',
            'test_year': test_year,
            'train_range': f'{train_start}-{train_end}',
            'stocks': {}
        }

        if not pkl_file.exists():
            all_rows.append(row)
            continue

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Use extract_window_metrics for extended metric support
        metrics = extract_window_metrics(data)
        row['stocks'] = metrics

        for t in STOCKS:
            s = metrics.get(t)
            if s is not None:
                stock_total[t] += 1
                if s['lesr_sharpe'] > s['base_sharpe']:
                    stock_wins[t] += 1
                    total_lesr_wins += 1
                elif s['base_sharpe'] > s['lesr_sharpe']:
                    total_base_wins += 1
                else:
                    total_ties += 1

        all_rows.append(row)

    total = total_lesr_wins + total_base_wins + total_ties
    if total > 0:
        print(f"总体 LESR 胜率: {total_lesr_wins}/{total} ({total_lesr_wins/total*100:.1f}%)")
        print(f"总体 Base 胜率: {total_base_wins}/{total} ({total_base_wins/total*100:.1f}%)")
        print()
        print("各股票 LESR 胜率:")
        for t in STOCKS:
            if stock_total[t] > 0:
                print(f"  {t}: {stock_wins[t]}/{stock_total[t]} ({stock_wins[t]/stock_total[t]*100:.1f}%)")

    # Generate markdown report
    report = generate_markdown_report(all_rows, stocks=STOCKS)
    md_file = EXP_DIR / "sliding_window_report_extended.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nMarkdown report saved: {md_file}")


if __name__ == '__main__':
    main()
