"""
Portfolio Environment for RL Training

Simulates portfolio management with 5 stocks + cash.
State: per-stock features (from feature_library) + portfolio features + regime + weights
Action: target weights (6 dims via softmax/Dirichlet)
Reward: mean-variance + reward rules

投资组合 RL 环境模块。

模拟 5 只股票（TSLA, NFLX, AMZN, MSFT, JNJ）+ 现金的投资组合管理。

状态向量布局：
  - 压缩原始状态：10 维 * 5 只股票 = 50 维
  - revise_state 额外特征：K 维 * 5 只股票
  - 组合级特征：P 维
  - 市场状态向量：3 维（趋势/波动率/风险）
  - 当前权重：6 维

动作空间：6 维目标权重（通过 Dirichlet 分布采样，自动归一化为权重和 = 1）

奖励计算：基础 Mean-Variance 奖励 + LLM 选择的奖励规则 + 内在奖励
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional

TICKERS = ['TSLA', 'NFLX', 'AMZN', 'MSFT', 'JNJ']
N_ASSETS = 6  # 5 stocks + cash
WINDOW = 20   # lookback window in trading days
STATE_CHANNELS = 6  # close, open, high, low, volume, adj_close per day


class PortfolioEnv:
    """RL environment for portfolio optimization.

    投资组合优化的 RL 环境。
    核心 API：reset() → step(weights) → 循环直到 done
    """

    def __init__(self, data_path: str, config: dict,
                 revise_state_fn: Callable = None,
                 portfolio_features_fn: Callable = None,
                 reward_rules_fn: Callable = None,
                 detect_regime_fn: Callable = None,
                 intrinsic_reward_fn: Callable = None,
                 train_period: tuple = None,
                 transaction_cost: float = 0.001):
        """Initialize environment.

        Args:
            data_path: path to pickle file from prepare_data.py
            config: config dict with keys from config.yaml
            revise_state_fn: closure from feature_library.build_revise_state()
            portfolio_features_fn: closure from portfolio_features.build_portfolio_features()
            reward_rules_fn: closure from reward_rules.build_reward_rules()
            detect_regime_fn: regime_detector.detect_market_regime
            train_period: (start_date, end_date) strings
            transaction_cost: transaction cost rate
        """
        with open(data_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        self.config = config
        self.revise_state_fn = revise_state_fn
        self.portfolio_features_fn = portfolio_features_fn
        self.reward_rules_fn = reward_rules_fn
        self.detect_regime_fn = detect_regime_fn
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.transaction_cost = transaction_cost

        # Sort dates
        self.all_dates = sorted(self.raw_data.keys())
        if train_period:
            self.dates = [d for d in self.all_dates
                          if train_period[0] <= d <= train_period[1]]
        else:
            self.dates = self.all_dates

        # Pre-extract price matrices
        self._build_price_matrix()

        # State tracking
        self.current_step = 0
        self.weights = np.ones(N_ASSETS) / N_ASSETS
        self.portfolio_value = 1.0
        self.peak_value = 1.0

    def _build_price_matrix(self):
        """Build aligned price matrix (T, 5) for each date.

        构建日期对齐的价格和成交量矩阵，用于快速查找。
        """
        self.prices = {}
        self.volumes = {}

        for date in self.dates:
            day_data = self.raw_data.get(date, {})
            price_row = {}
            vol_row = {}
            for ticker in TICKERS:
                info = day_data.get('price', {}).get(ticker, {})
                close = info.get('close', 0.0)
                adj_close = info.get('adjusted_close', close)
                vol = info.get('volume', 0.0)
                price_row[ticker] = adj_close if adj_close > 0 else close
                vol_row[ticker] = vol
            self.prices[date] = price_row
            self.volumes[date] = vol_row

    def _get_raw_state(self, ticker: str, date_idx: int) -> np.ndarray:
        """Build 120d interleaved raw state for a single ticker.

        Layout: [close, open, high, low, volume, adj_close] * 20 days

        构建单只股票的 120 维交错原始状态。
        布局：[收盘价, 开盘价, 最高价, 最低价, 成交量, 复权收盘价] * 20 天
        """
        start_idx = max(0, date_idx - WINDOW + 1)
        end_idx = date_idx + 1

        state = np.zeros(WINDOW * STATE_CHANNELS)
        for i, didx in enumerate(range(start_idx, end_idx)):
            if didx < 0 or didx >= len(self.dates):
                continue
            date = self.dates[didx]
            day_data = self.raw_data.get(date, {}).get('price', {}).get(ticker, {})
            offset = i * STATE_CHANNELS
            state[offset + 0] = day_data.get('close', 0.0)
            state[offset + 1] = day_data.get('open', 0.0)
            state[offset + 2] = day_data.get('high', 0.0)
            state[offset + 3] = day_data.get('low', 0.0)
            state[offset + 4] = day_data.get('volume', 0.0)
            state[offset + 5] = day_data.get('adjusted_close',
                                              day_data.get('close', 0.0))
        return state

    def _get_raw_states_dict(self, date_idx: int) -> Dict[str, np.ndarray]:
        """Get raw states for all tickers."""
        return {t: self._get_raw_state(t, date_idx) for t in TICKERS}

    def _compress_raw_state(self, raw_state: np.ndarray) -> np.ndarray:
        """Compress 120-dim raw state to ~10 dims: 5 recent closes + 5 recent returns.

        将 120 维原始状态压缩为约 10 维：5 个近期收盘价（归一化）+ 5 个近期收益率。
        这样可以在保留关键信息的同时大幅降低状态维度。
        """
        closes = np.array([raw_state[i * 6] for i in range(WINDOW)], dtype=float)
        mean_c = np.mean(closes) + 1e-10
        recent_closes = (closes[-5:] / mean_c) - 1.0
        returns = np.diff(closes) / (closes[:-1] + 1e-10)
        recent_returns = returns[-5:] if len(returns) >= 5 else np.zeros(5)
        return np.concatenate([recent_closes, recent_returns])

    def _compute_state(self, date_idx: int) -> np.ndarray:
        """Compute full observation vector.

        Layout:
          - Compressed raw: 10 dims * 5 stocks
          - Revised features (extras only): K dims * 5 stocks
          - Portfolio features: P dims
          - Regime vector: 3 dims
          - Current weights: 6 dims

        计算完整的观测向量。

        状态布局（各部分拼接）：
          - 压缩原始状态：10 维 * 5 只股票 = 50 维
          - revise_state 额外特征：K 维 * 5 只股票（LLM 生成的特征）
          - 组合级特征：P 维
          - 市场状态向量：3 维（趋势方向/波动率水平/风险水平）
          - 当前权重：6 维
        """
        raw_states = self._get_raw_states_dict(date_idx)

        parts = []

        # Compressed raw per stock (10 * 5 = 50)
        compressed = []
        for ticker in TICKERS:
            compressed.append(self._compress_raw_state(raw_states[ticker]))
        parts.append(np.concatenate(compressed))

        # Revised extras per stock (K * 5)
        if self.revise_state_fn:
            revised_per_stock = []
            for ticker in TICKERS:
                full_revised = self.revise_state_fn(raw_states[ticker])
                extras = full_revised[120:] if len(full_revised) > 120 else np.array([0.0])
                if np.any(np.isnan(extras)) or np.any(np.isinf(extras)):
                    extras = np.zeros_like(extras)
                revised_per_stock.append(extras)
            parts.append(np.concatenate(revised_per_stock))
        else:
            parts.append(np.zeros(5))

        # Portfolio features
        if self.portfolio_features_fn:
            port_feat = self.portfolio_features_fn(raw_states, self.weights)
            parts.append(port_feat)
        else:
            parts.append(np.zeros(5))

        # Regime
        if self.detect_regime_fn:
            regime = self.detect_regime_fn(raw_states)
        else:
            regime = np.array([0.0, 0.5, 0.0])
        parts.append(regime)

        # Current weights
        parts.append(self.weights)

        return np.concatenate(parts)

    @property
    def state_dim(self):
        """Compute state dimension by doing a dry run.

        通过一次试算来获取实际状态维度（因为维度取决于 LLM 生成的特征数）。
        """
        if not self.dates:
            return 100
        state = self._compute_state(0)
        return len(state)

    def reset(self) -> np.ndarray:
        """Reset environment to start of period.

        重置环境到训练期起始位置。
        权重初始化为等权（1/6），组合价值初始化为 1.0。
        """
        self.current_step = WINDOW  # need WINDOW days of history
        self.weights = np.ones(N_ASSETS) / N_ASSETS
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._compute_state(self.current_step)

    def step(self, target_weights: np.ndarray):
        """Execute one step.

        Args:
            target_weights: array of shape (6,), should sum to 1

        Returns:
            next_state, reward, done, info

        执行一步交易：
        1. 归一化目标权重
        2. 计算组合收益（含交易成本）
        3. 计算奖励 = 基础MV奖励 + 奖励规则 + 内在奖励
        4. 返回下一状态、奖励、是否结束、信息字典
        """
        # Normalize weights
        target_weights = np.clip(target_weights, 0, 1)
        total = target_weights.sum()
        if total > 1e-8:
            target_weights = target_weights / total
        else:
            target_weights = np.ones(N_ASSETS) / N_ASSETS

        prev_weights = self.weights.copy()
        self.current_step += 1

        if self.current_step >= len(self.dates):
            done = True
            return np.zeros(self.state_dim), 0.0, done, {}

        done = self.current_step >= len(self.dates) - 1

        # Compute portfolio return
        date = self.dates[self.current_step]
        prev_date = self.dates[self.current_step - 1]

        stock_returns = np.zeros(len(TICKERS))
        for i, ticker in enumerate(TICKERS):
            prev_price = self.prices.get(prev_date, {}).get(ticker, 0.0)
            curr_price = self.prices.get(date, {}).get(ticker, 0.0)
            if prev_price > 0:
                stock_returns[i] = (curr_price - prev_price) / prev_price
            else:
                stock_returns[i] = 0.0

        # Cash return (0 for simplicity, or risk-free rate)
        cash_return = 0.0
        all_returns = np.append(stock_returns, cash_return)

        # Portfolio return before transaction costs
        port_return = float(np.dot(prev_weights, all_returns))

        # Transaction cost
        turnover = float(np.sum(np.abs(target_weights - prev_weights))) / 2.0
        tc_cost = turnover * self.transaction_cost

        # Net return
        net_return = port_return - tc_cost

        # Update portfolio value
        self.portfolio_value *= (1 + net_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Update weights (drift + rebalance)
        drifted = prev_weights * (1 + all_returns)
        drifted_sum = drifted.sum()
        if drifted_sum > 1e-8:
            drifted = drifted / drifted_sum
        self.weights = target_weights

        # Compute reward
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Base reward: mean-variance
        lambda_mv = self.config.get('portfolio', {}).get('default_lambda', 0.5)
        base_reward = net_return - lambda_mv * (current_drawdown ** 2)

        # Additional reward rules
        rule_bonus = 0.0
        trigger_log = {}
        if self.reward_rules_fn:
            raw_states = self._get_raw_states_dict(self.current_step)
            regime = self.detect_regime_fn(raw_states) if self.detect_regime_fn else None
            port_feats = {}
            if self.portfolio_features_fn:
                pf_raw = self.portfolio_features_fn(raw_states, self.weights)
                port_feats['raw'] = pf_raw

            # Compute momentum_rank for momentum_alignment rule
            try:
                from core.portfolio_features import compute_momentum_rank
                port_feats['momentum_rank'] = compute_momentum_rank(raw_states, current_weights=self.weights)
            except Exception:
                pass

            rule_bonus, trigger_log = self.reward_rules_fn(
                weights=self.weights,
                prev_weights=prev_weights,
                regime_vector=regime,
                portfolio_features=port_feats,
                base_reward=base_reward,
                current_drawdown=current_drawdown,
            )

        # Intrinsic reward from LLM code (averaged across all tickers)
        intrinsic_r = 0.0
        if self.intrinsic_reward_fn and self.revise_state_fn:
            try:
                raw_states_now = self._get_raw_states_dict(self.current_step)
                ir_values = []
                for ticker in TICKERS:
                    revised = self.revise_state_fn(raw_states_now[ticker])
                    ir_values.append(float(self.intrinsic_reward_fn(revised)))
                intrinsic_r = float(np.mean(ir_values))
                intrinsic_r = np.clip(intrinsic_r, -1.0, 1.0)
            except Exception:
                intrinsic_r = 0.0

        reward = base_reward + rule_bonus + intrinsic_r

        # Next state
        next_state = self._compute_state(self.current_step)

        info = {
            'portfolio_return': net_return,
            'transaction_cost': tc_cost,
            'turnover': turnover,
            'drawdown': current_drawdown,
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'trigger_log': trigger_log,
            'intrinsic_reward': float(intrinsic_r),
        }

        return next_state, reward, done, info

    def get_revised_states(self, n_samples: int = 200) -> tuple:
        """Get revised states and forward returns for IC computation.

        获取修订后的状态和远期收益，用于 IC 计算。
        返回：(revised_states, forward_returns, regime_labels)
        """
        if self.revise_state_fn is None:
            return np.array([]), np.array([]), np.array([])

        n = min(n_samples, len(self.dates) - WINDOW - 1)
        if n < 10:
            return np.array([]), np.array([]), np.array([])
        indices = np.linspace(WINDOW, len(self.dates) - 2, n, dtype=int)

        revised_list = []
        forward_list = []
        regime_labels = []

        for idx in indices:
            raw_states = self._get_raw_states_dict(idx)
            revised = self.revise_state_fn(raw_states[TICKERS[0]])
            revised_list.append(revised)

            date = self.dates[idx]
            next_date = self.dates[idx + 1]
            ret = 0.0
            for ticker in TICKERS:
                p0 = self.prices.get(date, {}).get(ticker, 0.0)
                p1 = self.prices.get(next_date, {}).get(ticker, 0.0)
                if p0 > 0:
                    ret += (p1 - p0) / p0
            forward_list.append(ret / len(TICKERS))

            if self.detect_regime_fn:
                rv = self.detect_regime_fn(raw_states)
                from ic_analyzer import _classify_regime
                regime_labels.append(_classify_regime(rv[0], rv[1]))
            else:
                regime_labels.append('neutral')

        return np.array(revised_list), np.array(forward_list), np.array(regime_labels)

    def get_training_states(self, n_samples: int = 200) -> tuple:
        """Get sample states and forward returns for feature screening.

        Returns:
            training_states: dict {ticker: array of states}
            forward_returns: array of forward portfolio returns

        获取训练样本状态和远期收益，用于特征筛选和 LLM 提示词的市场统计注入。
        返回：{ticker: 状态数组} 和等权组合远期收益数组
        """
        n = min(n_samples, len(self.dates) - WINDOW - 1)
        indices = np.linspace(WINDOW, len(self.dates) - 2, n, dtype=int)

        training_states = {t: [] for t in TICKERS}
        forward_returns = []

        for idx in indices:
            for ticker in TICKERS:
                training_states[ticker].append(self._get_raw_state(ticker, idx))

            # Forward return: equal-weight portfolio
            date = self.dates[idx]
            next_date = self.dates[idx + 1]
            ret = 0.0
            for ticker in TICKERS:
                p0 = self.prices.get(date, {}).get(ticker, 0.0)
                p1 = self.prices.get(next_date, {}).get(ticker, 0.0)
                if p0 > 0:
                    ret += (p1 - p0) / p0
            forward_returns.append(ret / len(TICKERS))

        for ticker in TICKERS:
            training_states[ticker] = np.array(training_states[ticker])
        forward_returns = np.array(forward_returns)

        return training_states, forward_returns
