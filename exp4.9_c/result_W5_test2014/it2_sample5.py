import numpy as np

def revise_state(s):
    # Extract closing prices
    closing_prices = s[0::6]  # every 6th element starting from index 0
    
    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0

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

    # Feature 3: Average True Range (ATR) for volatility
    price_changes = np.diff(closing_prices)
    atr = np.mean(np.abs(price_changes[-14:])) if len(price_changes) >= 14 else 0

    # Feature 4: Bollinger Bands (20-day SMA and standard deviation)
    if len(closing_prices) >= 20:
        sma = np.mean(closing_prices[-20:])  # 20-day Simple Moving Average
        std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
    else:
        upper_band, lower_band = 0, 0

    # Feature 5: Price Change % (current close relative to the previous close)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0

    features = [price_momentum, rsi, atr, upper_band, lower_band, price_change_pct]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_std = np.std(enhanced_s[0:120])  # Standard deviation of historical prices
    price_change_threshold = historical_std
    rsi_threshold_high = 70
    rsi_threshold_low = 30

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > 0.7:
        reward += -40 if features[0] > 0 else 5  # Strong negative for BUY, mild positive for SELL
    elif risk_level > 0.4:
        reward += -20 if features[0] > 0 else 0  # Moderate negative for BUY

    # **Priority 2 — TREND FOLLOWING (when risk is low)**
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Positive reward for correct bullish signal
            reward += 15
        elif trend_direction < -0.3 and features[0] < 0:  # Positive reward for correct bearish signal
            reward += 15

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < rsi_threshold_low:  # Oversold condition
            reward += 10  # Reward for buying in mean-reversion condition
        elif features[1] > rsi_threshold_high:  # Overbought condition
            reward += 10  # Reward for selling in mean-reversion condition

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds