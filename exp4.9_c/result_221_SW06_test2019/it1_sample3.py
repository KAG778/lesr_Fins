import numpy as np

def revise_state(s):
    # Extracting necessary price information
    closing_prices = s[0:120:6]  # Closing prices
    high_prices = s[2:120:6]     # High prices
    low_prices = s[3:120:6]      # Low prices

    # Feature 1: Average True Range (ATR)
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 2: Bollinger Bands (upper and lower bands)
    sma = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0  # 20-day SMA
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0  # 20-day std dev
    upper_band = sma + (std_dev * 2)  # Upper Bollinger Band
    lower_band = sma - (std_dev * 2)  # Lower Bollinger Band
    bb_range = upper_band - lower_band if std_dev != 0 else 0

    # Feature 3: Exponential Moving Average (EMA)
    ema_period = 14
    ema = np.mean(closing_prices[-ema_period:]) if len(closing_prices) >= ema_period else 0  # Simple EMA calculation

    features = [atr, bb_range, ema]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # The new features
    reward = 0.0

    # Calculate relative thresholds based on historical std
    historical_std = np.std(features)
    low_threshold = np.mean(features) - (1 * historical_std)
    high_threshold = np.mean(features) + (1 * historical_std)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for high-risk BUY
        reward += np.random.uniform(5, 10)   # Mild positive for high-risk SELL
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 20)    # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[2] > high_threshold:  # Assuming feature[2] indicates bullish momentum
            reward += 10  # Reward for upward momentum
        elif features[2] < low_threshold:  # Assuming feature[2] indicates bearish momentum
            reward += 10  # Reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < low_threshold:  # Oversold condition, buy opportunity
            reward += 10  # Reward for buying oversold
        elif features[1] > high_threshold:  # Overbought condition, sell opportunity
            reward -= 10  # Penalty for buying overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce magnitude of reward by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds