import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    features = []

    # Feature 1: Price Change Ratio
    price_change_ratio = np.zeros(19)  # No change for the first day
    for i in range(1, 20):
        if closing_prices[i-1] != 0:  # Avoid division by zero
            price_change_ratio[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
        else:
            price_change_ratio[i-1] = 0  # Handle missing values

    features.append(np.mean(price_change_ratio))  # Mean price change ratio

    # Feature 2: Volume Change Ratio
    volume_change_ratio = np.zeros(19)  # No change for the first day
    for i in range(1, 20):
        if volumes[i-1] != 0:  # Avoid division by zero
            volume_change_ratio[i-1] = (volumes[i] - volumes[i-1]) / volumes[i-1]
        else:
            volume_change_ratio[i-1] = 0  # Handle missing values

    features.append(np.mean(volume_change_ratio))  # Mean volume change ratio

    # Feature 3: MACD
    short_window = 12
    long_window = 26
    signal_window = 9

    if len(closing_prices) >= long_window:
        short_ema = np.mean(closing_prices[-short_window:])
        long_ema = np.mean(closing_prices[-long_window:])
        macd = short_ema - long_ema
        signal = np.mean(closing_prices[-signal_window:])
        features.append(macd - signal)  # MACD - Signal Line
    else:
        features.append(0)  # Not enough data to calculate MACD

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]  # trend_direction
    volatility_level = regime[1]  # volatility_level
    risk_level = regime[2]        # risk_level

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(reward)