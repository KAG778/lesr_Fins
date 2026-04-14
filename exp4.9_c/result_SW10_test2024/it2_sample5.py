import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # 20 days closing prices
    volumes = s[4:120:6]          # 20 days volumes
    
    features = []

    # 1. Downside Volatility (semivariance of returns) over last 20 days to capture crisis risk
    returns = np.diff(closing_prices) / closing_prices[:-1]
    downside_returns = returns[returns < 0]
    downside_vol = np.sqrt(np.mean(downside_returns**2)) if downside_returns.size > 0 else 0
    features.append(downside_vol)

    # 2. Price drawdown from 20-day peak (%), to quantify recent crisis-like conditions
    peak_price = np.max(closing_prices)
    current_price = closing_prices[-1]
    drawdown_pct = (peak_price - current_price) / peak_price if peak_price != 0 else 0
    features.append(drawdown_pct)

    # 3. Volume Spike Indicator: current volume relative to 20-day median volume
    median_vol = np.median(volumes) if len(volumes) > 0 else 1
    volume_spike = (volumes[-1] - median_vol) / median_vol if median_vol != 0 else 0
    features.append(volume_spike)

    # 4. Trend Strength via ADX (Average Directional Index) proxy:
    # Since no highs/lows, approximate ADX by normalized absolute price changes ratio over 14 days
    period = min(14, len(closing_prices)-1)
    if period > 0:
        abs_changes = np.abs(np.diff(closing_prices[-(period+1):]))
        sum_abs_changes = np.sum(abs_changes)
        net_change = np.abs(closing_prices[-1] - closing_prices[-(period+1)])
        adx_proxy = net_change / sum_abs_changes if sum_abs_changes != 0 else 0
    else:
        adx_proxy = 0
    features.append(adx_proxy)

    # 5. Mean Reversion Signal: distance from 10-day moving average normalized by std dev
    ma_period = min(10, len(closing_prices))
    ma_10 = np.mean(closing_prices[-ma_period:])
    std_10 = np.std(closing_prices[-ma_period:]) if ma_period > 1 else 1
    mean_rev_signal = (current_price - ma_10) / std_10 if std_10 != 0 else 0
    features.append(mean_rev_signal)

    return np.array(features)


def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Use historical std of feature 0 (downside volatility) and feature 1 (drawdown) for adaptive thresholds
    # If not enough data, fallback to fixed small values to avoid division errors
    hist_downside_vol_std = np.std(features[0]) if np.ndim(features) > 0 else 0.01
    hist_drawdown_std = np.std(features[1]) if np.ndim(features) > 0 else 0.01

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features (momentum/positive trend)
        # Mild positive reward for SELL-aligned features (negative trend or mean reversion signals)
        reward -= 40
        if trend_direction < 0:
            reward += 10
        # Additionally penalize if crisis drawdown feature is large (feature[1])
        if features[1] > (2 * hist_drawdown_std):
            reward -= 10

    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    else:
        # Priority 2: TREND FOLLOWING (when risk is low)
        if abs(trend_direction) > 0.3:
            # Align reward with trend and trend strength proxy (feature 3 = adx_proxy)
            adx_proxy = features[3] if len(features) > 3 else 0
            trend_strength = min(max(adx_proxy, 0), 1)  # clamp to [0,1]
            if trend_direction > 0:
                reward += 25 * trend_strength
            else:
                reward += 20 * trend_strength

        # Priority 3: SIDEWAYS / MEAN REVERSION (low trend magnitude and low risk)
        elif abs(trend_direction) <= 0.3 and risk_level < 0.3:
            # Reward mean reversion feature (feature 4), positive if price below MA (negative value)
            mean_rev_signal = features[4] if len(features) > 4 else 0
            # If price is significantly below MA (mean_rev_signal < -1), reward buy signals
            if mean_rev_signal < -1:
                reward += 20
            # If price is significantly above MA (mean_rev_signal > 1), reward sell signals
            elif mean_rev_signal > 1:
                reward += 20
            else:
                reward += 5  # Mild reward for mild mean reversion

    # Priority 4: HIGH VOLATILITY
    # Use relative threshold for volatility_level based on historical downside volatility std dev
    vol_thresh = np.mean(features[0]) + 1.5 * hist_downside_vol_std if len(features) > 0 else 0.5
    if volatility_level > vol_thresh and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip reward to required range
    reward = np.clip(reward, -100, 100)

    return reward