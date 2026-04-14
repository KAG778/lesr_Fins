import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Dynamic Moving Average based on recent volatility (last 20 days)
    if len(closing_prices) >= 20:
        dynamic_ma = np.mean(closing_prices[-20:])
    else:
        dynamic_ma = closing_prices[-1] if len(closing_prices) > 0 else 0

    # Feature 2: Average True Range (ATR) over the last 14 days
    true_ranges = np.abs(np.diff(closing_prices, prepend=closing_prices[0]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.mean(true_ranges)

    # Feature 3: Bollinger Bands (20 days)
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
    else:
        upper_band = lower_band = closing_prices[-1]  # Fallback to last price

    # Feature 4: Cumulative Returns over the last 20 days
    cum_return = (closing_prices[-1] / closing_prices[-21] - 1) if len(closing_prices) > 20 else 0

    # Combine features
    features = [dynamic_ma, atr, upper_band, lower_band, cum_return]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    # Calculate relative thresholds for risk management
    risk_threshold_high = 0.7
    risk_threshold_moderate = 0.4
    volatility_threshold_high = 0.6

    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Expectation of upward momentum
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # Expectation of downward momentum
            reward += np.random.uniform(5, 10)

    elif risk_level > risk_threshold_moderate:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Overbought condition
            reward += np.random.uniform(10, 20)
        elif features[0] > 0.1:  # Oversold condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold_high and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))