import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volume = s[4::6]          # Extract volume

    # Feature 1: Price Momentum (last 10 days)
    price_momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) >= 11 else 0

    # Feature 2: Relative Strength Index (RSI) - calculated over last 14 periods
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: Average True Range (ATR) - volatility measure
    true_ranges = high_prices[-14:] - low_prices[-14:]  # ATR over last 14 days
    atr = np.mean(true_ranges) if len(true_ranges) >= 14 else 0

    # Feature 4: Volume Change (percentage change over 10 days)
    volume_change = (volume[-1] - volume[-11]) / volume[-11] if volume[-11] != 0 else 0

    # Feature 5: Exponential Moving Average (EMA) - 10-day EMA
    ema = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0

    features = [price_momentum, rsi, atr, volume_change, ema]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical standard deviation of risk levels
    risk_std = np.std([0.1, 0.4, 0.7])  # Placeholder, replace with actual historical data
    low_risk_threshold = 0.4 * risk_std
    high_risk_threshold = 0.7 * risk_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals in high risk
    elif risk_level > low_risk_threshold:
        reward += 20  # Mildly positive reward for SELL signals in moderate risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 10 * trend_direction  # Reward for positive momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 10 * -trend_direction  # Reward for negative momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        rsi = enhanced_s[123][1]  # Get RSI from features
        if rsi < 30:  # Oversold
            reward += 15  # Reward for mean-reversion buy signal
        elif rsi > 70:  # Overbought
            reward += 15  # Reward for mean-reversion sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip reward to be within [-100, 100]
    return float(np.clip(reward, -100, 100))