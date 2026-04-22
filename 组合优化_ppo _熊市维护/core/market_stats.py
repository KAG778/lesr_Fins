"""
Market Statistics for LLM Prompt Injection

Pre-computes per-stock stats, correlation matrix, and regime summary from
training data. Every number includes an interpretation string.

市场统计模块 —— 用于 LLM 提示词注入。

预计算以下市场信息并格式化为文本，注入到 LLM 提示词中：
1. 每只股票的概况（行业、日波动率、20日收益率、解读）
2. 股票间的相关性矩阵
3. 分散化建议（基于相关性）

关键设计：每个统计数字都附带解读字符串，帮助 LLM 理解数据含义。
这些统计数据注入到 init_prompt 和 next_iteration_prompt 的 Market Statistics 部分。
"""

import numpy as np

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']
TICKER_PROFILES = {
    'TSLA': {'sector': 'EV/Tech', 'vol_profile': 'Very high (~4% daily)'},
    'NFLX': {'sector': 'Streaming', 'vol_profile': 'Medium-high (~2.5% daily)'},
    'AMZN': {'sector': 'E-commerce/Cloud', 'vol_profile': 'Medium (~2.2% daily)'},
    'MSFT': {'sector': 'Software', 'vol_profile': 'Low-medium (~1.8% daily)'},
    'JNJ': {'sector': 'Pharma', 'vol_profile': 'Low (~1.2% daily, defensive)'},
}


def _extract_closes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6] for i in range(n)], dtype=float)


def _extract_volumes(s: np.ndarray) -> np.ndarray:
    n = len(s) // 6
    return np.array([s[i * 6 + 4] for i in range(n)], dtype=float)


def get_market_stats(training_states: dict) -> str:
    """Compute full market statistics with interpretation for LLM prompt.

    Args:
        training_states: dict {ticker: array_of_120d_states (N, 120)} or
                         dict {ticker: 120d_raw_state} for single snapshot.

    Returns:
        Formatted string with tables, correlation matrix, and interpretation.

    计算完整的市场统计数据并格式化为 LLM 提示词文本。
    输出包含：
    1. 每只股票的概况表格（行业、日波动率、20日收益、解读）
    2. 5x5 相关性矩阵
    3. 平均相关性和分散化建议

    参数：
        training_states: 字典 {ticker: 120维状态数组 或 (N, 120) 状态矩阵}

    返回：
        格式化的 Markdown 文本，直接注入 LLM 提示词
    """
    lines = []

    # Per-stock stats
    lines.append("### Per-Stock Profile")
    lines.append("| Ticker | Sector | Daily Vol | 20d Return | Interpretation |")
    lines.append("|--------|--------|-----------|------------|----------------|")

    for ticker in TICKERS:
        states = training_states.get(ticker)
        if states is None:
            continue
        if states.ndim == 1:
            states = states.reshape(1, -1)

        closes_list = []
        vols_list = []
        for s in states:
            c = _extract_closes(s)
            closes_list.append(c)
            vols_list.append(_extract_volumes(s))

        all_closes = np.concatenate(closes_list) if closes_list else np.array([100.0])
        all_vols = np.concatenate(vols_list) if vols_list else np.array([1e6])

        returns = np.diff(all_closes) / (all_closes[:-1] + 1e-10)
        daily_vol = float(np.std(returns)) * 100 if len(returns) > 1 else 0.0

        ret_20d = 0.0
        if len(all_closes) >= 21 and all_closes[-21] != 0:
            ret_20d = (all_closes[-1] - all_closes[-21]) / abs(all_closes[-21]) * 100

        avg_vol = float(np.mean(all_vols))

        profile = TICKER_PROFILES.get(ticker, {})
        sector = profile.get('sector', 'Unknown')
        vol_desc = profile.get('vol_profile', 'Unknown')

        # Interpretation
        if daily_vol > 3.0:
            interp = f"High vol ({vol_desc}). Good for trend-following, but risky. Consider concentration limits."
        elif daily_vol > 2.0:
            interp = f"Medium vol ({vol_desc}). Balanced risk/reward. Core holding candidate."
        elif daily_vol > 1.5:
            interp = f"Lower vol ({vol_desc}). Stable performer. Good anchor stock."
        else:
            interp = f"Low vol ({vol_desc}). Defensive. Hedge against downturns. Useful when risk_level is high."

        if ret_20d > 3:
            interp += " Strong recent momentum."
        elif ret_20d < -3:
            interp += " Recent weakness -- consider reducing weight."

        lines.append(f"| {ticker} | {sector} | {daily_vol:.1f}% | {ret_20d:+.1f}% | {interp} |")

    lines.append("")

    # Correlation matrix
    lines.append("### Correlation Matrix (20-day rolling returns)")
    all_returns = {}
    for ticker in TICKERS:
        states = training_states.get(ticker)
        if states is None:
            continue
        if states.ndim == 1:
            states = states.reshape(1, -1)
        closes_list = [np.concatenate([_extract_closes(s) for s in states])]
        all_c = closes_list[0]
        if len(all_c) > 1:
            all_returns[ticker] = np.diff(all_c) / (all_c[:-1] + 1e-10)

    if len(all_returns) == 5:
        header = "        " + "  ".join(f"{t:>5s}" for t in TICKERS)
        lines.append(header)
        pair_corrs = {}
        for i, t1 in enumerate(TICKERS):
            row = f"{t1:>5s}   "
            for j, t2 in enumerate(TICKERS):
                if i == j:
                    row += "  1.00"
                elif t1 in all_returns and t2 in all_returns:
                    r1, r2 = all_returns[t1], all_returns[t2]
                    n = min(len(r1), len(r2))
                    if n > 5:
                        c = np.corrcoef(r1[-n:], r2[-n:])[0, 1]
                        c = 0.0 if np.isnan(c) else c
                    else:
                        c = 0.0
                    row += f"  {c:.2f}"
                    if i < j:
                        pair_corrs[(t1, t2)] = c
                else:
                    row += "   N/A"
            lines.append(row)

        avg_corr = np.mean(list(pair_corrs.values())) if pair_corrs else 0.0
        lines.append(f"\nAverage pairwise correlation: {avg_corr:.2f}")
        if avg_corr < 0.3:
            lines.append("-> Low correlation: excellent diversification opportunity")
        elif avg_corr < 0.5:
            lines.append("-> Moderate correlation: some diversification benefit exists")
        else:
            lines.append("-> High correlation: limited diversification -- consider sector-exposure rules")

        if pair_corrs:
            min_pair = min(pair_corrs, key=pair_corrs.get)
            max_pair = max(pair_corrs, key=pair_corrs.get)
            lines.append(f"Lowest pair: {min_pair[0]}-{min_pair[1]} ({pair_corrs[min_pair]:.2f}) -> Most diversification value")
            lines.append(f"Highest pair: {max_pair[0]}-{max_pair[1]} ({pair_corrs[max_pair]:.2f}) -> Limited diversification between them")
        lines.append("")

    return "\n".join(lines)
