import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Price Momentum (current close - close from 6 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 6 else 0

    # Feature 2: Relative Strength Index (RSI) over the last 14 days
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices)

    # Feature 3: Bollinger Bands (20-day SMA and standard deviation)
    if len(closing_prices) >= 20:
        sma = np.mean(closing_prices[-20:])  # 20-day Simple Moving Average
        std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
    else:
        upper_band, lower_band = 0, 0

    # Feature 4: Average True Range (ATR) for volatility
    atr = np.mean(np.abs(np.diff(closing_prices[-14:]))) if len(closing_prices) > 14 else 0

    features = [price_momentum, rsi, upper_band, lower_band, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical thresholds for dynamic risk assessment
    historical_std = np.std(enhanced_s[0:120])  # Standard deviation of historical prices
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > risk_threshold_high:
        reward += -40 if features[0] > 0 else 5  # Strong negative for bullish features, mild positive for bearish features
    elif risk_level > risk_threshold_medium:
        reward += -20 if features[0] > 0 else 0  # Moderate negative for bullish features

    # **Priority 2 — TREND FOLLOWING (when risk is low)**
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3 and features[0] > 0:  # Positive reward for aligning with bullish momentum
            reward += 15
        elif trend_direction < -0.3 and features[0] < 0:  # Positive reward for aligning with bearish momentum
            reward += 15

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold (RSI)
            reward += 10  # Reward for buying in oversold conditions
        elif features[1] > 70:  # Overbought (RSI)
            reward += 10  # Reward for selling in overbought conditions

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds