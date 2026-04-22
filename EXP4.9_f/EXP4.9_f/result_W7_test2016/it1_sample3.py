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

    # Feature 3: MACD (Moving Average Convergence Divergence)
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

    # Feature 4: Relative Strength Index (RSI) for momentum assessment
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 5: Bollinger Bands (Upper and Lower)
    if len(closing_prices) >= 20:
        sma = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        features.append(upper_band)
        features.append(lower_band)
    else:
        features.extend([0, 0])  # Not enough data to calculate Bollinger Bands

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Define thresholds based on historical std
    risk_threshold = 0.5  # Example threshold for defining high risk
    trend_threshold = 0.3  # Example threshold for defining strong trend
    volatility_threshold = 0.6  # Example threshold for defining high volatility

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= 50  # Strong penalty for buying when risk is high
        reward += 10 if enhanced_s[123][0] < 0 else 0  # Mild positive for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 20 if enhanced_s[123][0] > 0 else 0
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:
            reward += 20 if enhanced_s[123][0] > 0 else 0  # Positive for upward features
        elif trend_direction < -trend_threshold:
            reward += 20 if enhanced_s[123][0] < 0 else 0  # Positive for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10 if enhanced_s[123][0] < 0 else 0  # Reward buying in mean reversion
        reward -= 10 if enhanced_s[123][0] > 0 else 0  # Penalize selling in mean reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure the reward is within the specified range
    reward = np.clip(reward, -100, 100)

    return float(reward)