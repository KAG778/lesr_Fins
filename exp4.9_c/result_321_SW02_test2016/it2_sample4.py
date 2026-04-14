import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0:120:6]  # Extract every 6th element for closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: 10-day Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0
     
    # Feature 2: Exponential Moving Average (EMA) of closing prices (last 10 days)
    weights = np.exp(np.linspace(-1., 0., 10))
    weights /= weights.sum()
    ema = np.dot(weights, closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    
    # Feature 3: Bollinger Bands - Distance from Upper and Lower Band
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_avg + (2 * std_dev)
        lower_band = moving_avg - (2 * std_dev)
        distance_from_upper = (closing_prices[-1] - upper_band) / (upper_band - lower_band) if upper_band != lower_band else 0
        distance_from_lower = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0
    else:
        distance_from_upper, distance_from_lower = 0, 0

    # Feature 4: Volume Oscillator (current vs. average volume over the last 10 days)
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 1.0
    volume_oscillator = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    # Combine features
    features = [price_momentum, ema, distance_from_upper, distance_from_lower, volume_oscillator]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    mean_risk = 0.5  # Placeholder for historical mean risk level
    std_risk = 0.2   # Placeholder for historical std for risk level
    risk_threshold = mean_risk + 1 * std_risk  # Example threshold based on std deviation
    trend_threshold = 0.3  # Placeholder for historical trend threshold

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for BUY when risk is high
        reward += np.random.uniform(5, 10)  # Mild positive for SELL
        return np.clip(reward, -100, 100)  # Early exit

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # Reward for bullish momentum
        else:
            reward += 20  # Reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion strategies

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]