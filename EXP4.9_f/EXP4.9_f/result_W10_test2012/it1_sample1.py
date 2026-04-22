import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Momentum (latest closing price - closing price 10 days ago)
    price_momentum = closing_prices[0] - closing_prices[10] if len(closing_prices) > 10 else 0

    # Feature 2: Relative Strength Index (RSI) Calculation
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Get the price changes
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if not enough data

    # Feature 3: Volume Change (percentage change from previous day)
    volume_change = (volumes[0] - volumes[1]) / volumes[1] * 100 if volumes[1] > 0 else 0

    # Feature 4: Price Volatility (standard deviation of the last 10 price changes)
    if len(closing_prices) > 10:
        price_changes = np.diff(closing_prices[-10:])
        price_volatility = np.std(price_changes)
    else:
        price_volatility = 0

    # Feature 5: Volume Volatility (standard deviation of the last 10 volume changes)
    if len(volumes) > 10:
        volume_changes = np.diff(volumes[-10:])
        volume_volatility = np.std(volume_changes)
    else:
        volume_volatility = 0

    features = [price_momentum, rsi, volume_change, price_volatility, volume_volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Dynamic thresholds based on historical data
    risk_threshold_high = 0.7
    risk_threshold_mid = 0.4
    trend_threshold = 0.3
    mean_rsi = 50
    rsi_threshold_low = 30
    rsi_threshold_high = 70

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_mid:
        reward -= np.random.uniform(10, 20)   # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_mid:
        if trend_direction > trend_threshold:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < risk_threshold_low:
        # Reward for mean-reversion features based on RSI
        if mean_rsi < rsi_threshold_low:
            reward += 15  # Reward for oversold
        elif mean_rsi > rsi_threshold_high:
            reward += 15  # Reward for overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]