import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    high_prices = s[2::6]      # Extracting high prices
    low_prices = s[3::6]       # Extracting low prices
    volumes = s[4::6]          # Extracting trading volumes

    # Feature 1: Price Momentum - difference between the most recent closing price and the closing price 5 days ago
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average True Range (ATR) over the last 14 days
    true_ranges = high_prices - low_prices
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Feature 3: Relative Strength Index (RSI) - calculated over the last 14 periods
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 4: Volume Spike - percentage change in volume compared to the average of the last 14 days
    avg_volume = np.mean(volumes[-14:]) if len(volumes) >= 14 else 1  # Avoid division by zero
    volume_change = (volumes[-1] - avg_volume) / avg_volume

    # Feature 5: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:])  # 12-period EMA
    long_ema = np.mean(closing_prices[-26:])   # 26-period EMA
    macd = short_ema - long_ema

    features = [price_momentum, atr, rsi, volume_change, macd]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds using standard deviation
    historical_std = np.std(enhanced_s[123:])  # Assuming features hold values that can be standardized
    price_momentum_threshold = historical_std * 0.5
    rsi_overbought_threshold = 70
    rsi_oversold_threshold = 30

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for BUY signals in high-risk environments
        reward += 20 if trend_direction < 0 else 0  # Mild positive for SELL signals
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += min(20, enhanced_s[123][0] / price_momentum_threshold)  # Scale reward by momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += min(20, -enhanced_s[123][0] / price_momentum_threshold)  # Scale reward for shorting

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123][2] < rsi_oversold_threshold:  # Oversold
            reward += 15  # Strong buy signal
        elif enhanced_s[123][2] > rsi_overbought_threshold:  # Overbought
            reward += 15  # Strong sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified bounds