import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    high_prices = s[2::6]      # Extracting high prices
    low_prices = s[3::6]       # Extracting low prices
    volumes = s[4::6]          # Extracting trading volumes

    # Feature 1: Price Momentum - difference between the most recent closing price and the closing price 5 days ago
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average True Range (ATR)
    # ATR = (High - Low) for each day, averaged over the last 14 days
    true_ranges = high_prices - low_prices
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Feature 3: Relative Strength Index (RSI) - calculated over 14 periods
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 4: Volume Oscillator (5-day EMA - 20-day EMA)
    ema_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    ema_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    volume_oscillator = ema_5 - ema_20

    features = [price_momentum, atr, rsi, volume_oscillator]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative for BUY signals
    elif risk_level > 0.4:
        reward += 20  # Mild positive for SELL signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Introduce thresholds based on historical standard deviations to adapt to different regimes
        historical_std = np.std(enhanced_s[123:])  # Use the features as the basis for historical std
        momentum_threshold = historical_std * 1.5  # 1.5 std dev as a relative threshold
        if trend_direction > 0.3 and enhanced_s[123][0] > momentum_threshold:  # Uptrend with positive momentum
            reward += 30  # Strong positive reward
        elif trend_direction < -0.3 and enhanced_s[123][0] < -momentum_threshold:  # Downtrend with negative momentum
            reward += 30  # Strong positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123][2] < 30:  # Oversold
            reward += 20  # Reward for buy signal
        elif enhanced_s[123][2] > 70:  # Overbought
            reward += 20  # Reward for sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified bounds