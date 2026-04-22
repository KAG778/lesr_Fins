import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]      # Extract high prices
    low_prices = s[3::6]       # Extract low prices
    volume = s[4::6]           # Extract volume

    # Feature 1: Price Momentum (last 5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average True Range (ATR) - volatility measure
    true_ranges = high_prices[-14:] - low_prices[-14:]  # ATR over last 14 days
    atr = np.mean(true_ranges) if len(true_ranges) >= 14 else 0

    # Feature 3: Relative Strength Index (RSI) calculated over last 14 periods
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 4: Volume Weighted Average Price (VWAP) - trend confirmation
    vwap = np.sum(closing_prices[-14:] * volume[-14:]) / np.sum(volume[-14:]) if np.sum(volume[-14:]) != 0 else 0

    features = [price_momentum, atr, rsi, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Determine thresholds based on historical standard deviation of risk_level
    risk_std = np.std(enhanced_s[120:123])  # Assuming the regime vector is updated over time
    high_risk_threshold = 0.7 * risk_std
    medium_risk_threshold = 0.4 * risk_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY signals
    elif risk_level > medium_risk_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        features = enhanced_s[123:]  # Get the features
        if trend_direction > 0.3:  # Uptrend
            reward += features[0]  # Positive momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += -features[0]  # Negative momentum (shorting)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123][2] < 30:  # Oversold
            reward += 10  # Reward for mean-reversion buy signal
        elif enhanced_s[123][2] > 70:  # Overbought
            reward += 10  # Reward for mean-reversion sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within specified bounds