import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volume = s[4::6]          # Extract volume

    # Feature 1: Price Momentum (last 5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average True Range (ATR)
    true_ranges = high_prices[-14:] - low_prices[-14:]
    atr = np.mean(true_ranges) if len(true_ranges) >= 14 else 0

    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 4: Volume Spike (current volume vs. average volume over last 20 days)
    avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else 0
    volume_spike = (volume[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    # Feature 5: Price Position within Bollinger Bands
    mean_price = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1  # Avoid division by zero
    upper_band = mean_price + (2 * std_dev)
    lower_band = mean_price - (2 * std_dev)
    price_position = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if upper_band - lower_band != 0 else 0

    features = [price_momentum, atr, rsi, volume_spike, price_position]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for risk levels
    historical_std = np.std([0.1, 0.4, 0.7])
    low_risk_threshold = 0.4 * historical_std
    high_risk_threshold = 0.7 * historical_std

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals in high risk
    elif risk_level > low_risk_threshold:
        reward += 20  # Mildly positive reward for SELL signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Strong uptrend
            reward += 10 * trend_direction  # Reward for positive momentum
        elif trend_direction < -0.3:  # Strong downtrend
            reward += 10 * -trend_direction  # Reward for negative momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        if enhanced_s[123][2] < 30:  # Oversold
            reward += 15  # Buy signal
        elif enhanced_s[123][2] > 70:  # Overbought
            reward += 15  # Sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within specified bounds