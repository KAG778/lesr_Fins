import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price momentum (current closing price vs. moving average of last 10 days)
    moving_average = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average) / (moving_average if moving_average != 0 else 1)

    # Feature 2: Average True Range (ATR) for volatility measure
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR

    # Feature 3: Distance from the 20-day moving average
    if len(closing_prices) >= 20:
        ma_20 = np.mean(closing_prices[-20:])
        distance_from_ma = (closing_prices[-1] - ma_20) / (ma_20 if ma_20 != 0 else 1)
    else:
        distance_from_ma = 0

    # Feature 4: Rate of Change (Momentum) over the last 14 days
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) > 15 else 0

    return np.array([price_momentum, atr, distance_from_ma, roc])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract computed features
    price_momentum = features[0]
    atr = features[1]
    distance_from_ma = features[2]
    roc = features[3]

    reward = 0.0

    # Calculate historical thresholds for risk management
    risk_threshold_high = 0.7  # This can be adjusted based on historical data
    risk_threshold_low = np.clip(0.4 + 0.1 * np.std([risk_level]), 0, 1)  # Relative based on historical std
    trend_threshold = np.clip(0.3 + 0.1 * np.std([trend_direction]), 0, 1)  # Relative based on historical std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if price_momentum > 0:  # Expectation of upward momentum (BUY-aligned)
            reward -= np.random.uniform(30, 50)  # Strong penalty
        else:  # Expectation of downward momentum (SELL-aligned)
            reward += np.random.uniform(5, 10)  # Mild positive reward

    elif risk_level > risk_threshold_low:
        if price_momentum > 0:
            reward -= np.random.uniform(10, 20)  # Moderate penalty for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_low:
        if trend_direction > trend_threshold and price_momentum > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -trend_threshold and price_momentum < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if distance_from_ma < -0.1:  # Oversold condition
            reward += np.random.uniform(10, 20)
        elif distance_from_ma > 0.1:  # Overbought condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_low:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]