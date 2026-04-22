import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    high_prices = s[2::6]      # Extracting high prices
    low_prices = s[3::6]       # Extracting low prices
    volumes = s[4::6]          # Extracting trading volumes

    # Feature 1: Price Momentum - difference between the most recent closing price and the closing price 5 days ago
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average True Range (ATR)
    true_ranges = high_prices - low_prices
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Feature 3: Relative Strength Index (RSI) - typically calculated over 14 periods
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 4: Bollinger Bands (Upper and Lower Bands)
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1  # Avoid division by zero
    upper_band = moving_avg + (2 * std_dev)
    lower_band = moving_avg - (2 * std_dev)

    # Feature 5: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices[-20:] * volumes[-20:]) / np.sum(volumes[-20:]) if np.sum(volumes[-20:]) != 0 else 0

    features = [price_momentum, atr, rsi, upper_band, lower_band, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_std = np.std([0.1, 0.4, 0.7])  # Placeholder for dynamic calculation
    low_risk_threshold = 0.4 * historical_std
    high_risk_threshold = 0.7 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for BUY signals in high risk
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals in moderate risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += np.clip(10 * trend_direction, 0, 20)  # Momentum reward
        elif trend_direction < -0.3:  # Downtrend
            reward += np.clip(10 * -trend_direction, 0, 20)  # Momentum reward for shorting

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        if enhanced_s[123][2] < 30:  # Oversold
            reward += 15  # Buy signal
        elif enhanced_s[123][2] > 70:  # Overbought
            reward += 15  # Sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip reward to be within [-100, 100]
    return float(np.clip(reward, -100, 100))