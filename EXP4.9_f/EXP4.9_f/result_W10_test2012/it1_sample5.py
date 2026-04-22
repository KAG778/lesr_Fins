import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Momentum (latest closing price - closing price 10 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-10] if len(closing_prices) > 10 else 0

    # Feature 2: Relative Strength Index (RSI)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if we don't have enough data

    # Feature 3: Volume Change (percentage change from previous day)
    volume_change = (volumes[0] - volumes[1]) / volumes[1] * 100 if volumes[1] > 0 else 0

    # Feature 4: Average True Range (ATR) for volatility
    if len(closing_prices) >= 14:
        high_low = np.max(closing_prices[-14:]) - np.min(closing_prices[-14:])
        atr = high_low / np.sqrt(14)  # Simplified ATR calculation
    else:
        atr = 0

    # Feature 5: Percentage Change from Moving Average (20-day MA)
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
        pct_change_from_ma = (closing_prices[-1] - moving_average) / moving_average * 100
    else:
        pct_change_from_ma = 0

    features = [price_momentum, rsi, volume_change, atr, pct_change_from_ma]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Define relative thresholds based on historical std deviation
    risk_threshold_high = 0.7
    risk_threshold_mid = 0.4
    volatility_threshold = 0.6

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)  # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_mid:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_mid:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward alignment
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(np.clip(reward, -100, 100))