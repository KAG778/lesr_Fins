import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # 1. Price Momentum (latest closing price - closing price 10 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0

    # 2. Relative Strength Index (RSI) calculation
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Get the price changes
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if we don't have enough data

    # 3. Average True Range (ATR) as a measure of volatility
    if len(closing_prices) > 14:
        high_low = np.array([s[i] - s[i + 1] for i in range(0, len(s) - 1, 6)])
        high_close = np.abs(high_low)  # Simplified ATR for demonstration
        atr = np.mean(high_close[-14:]) if len(high_close) >= 14 else 0
    else:
        atr = 0

    # 4. Bollinger Band Width (using a rolling mean and std)
    rolling_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bb_width = (rolling_mean + 2 * rolling_std) - (rolling_mean - 2 * rolling_std) if rolling_std > 0 else 0

    features = [price_momentum, rsi, atr, bb_width]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical std
    dynamic_threshold_high_risk = 0.7
    dynamic_threshold_moderate_risk = 0.4
    dynamic_threshold_trend = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level >= dynamic_threshold_high_risk:
        reward -= np.random.uniform(40, 60)  # Strong negative for BUY signals
        reward += np.random.uniform(5, 15)   # Mild positive for SELL signals
    elif risk_level >= dynamic_threshold_moderate_risk:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > dynamic_threshold_trend and risk_level < dynamic_threshold_moderate_risk:
        if trend_direction > dynamic_threshold_trend:
            reward += 20  # Positive reward for upward features
        elif trend_direction < -dynamic_threshold_trend:
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < dynamic_threshold_trend and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < dynamic_threshold_moderate_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]