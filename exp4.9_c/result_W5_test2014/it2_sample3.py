import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # every 6th element starting from index 0
    volumes = s[4::6]         # every 6th element starting from index 4

    # Feature 1: Price Change (%)
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0

    # Feature 2: Average Volume (last 20 days)
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0

    # Feature 3: Price Momentum (5-day)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 4: Relative Strength Index (RSI) over the last 14 days
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices)

    # Feature 5: Average True Range (ATR) for volatility (14-day)
    atr = np.mean(np.abs(np.diff(closing_prices[-14:]))) if len(closing_prices) > 14 else 0

    # Feature 6: Crisis Indicator (standard deviation of the last 20 days)
    crisis_indicator = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0

    features = [price_change, average_volume, price_momentum, rsi, atr, crisis_indicator]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate historical thresholds for dynamic risk assessment
    historical_std = np.std(enhanced_s[0:120])  # Standard deviation of historical prices
    price_change_threshold = np.mean(features[0]) + 1.5 * historical_std
    rsi_threshold_low = 30  # Typically oversold
    rsi_threshold_high = 70  # Typically overbought

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > 0.7:
        reward += -40 if features[0] > price_change_threshold else 5  # Strong negative for bullish features, mild positive for bearish features
    elif risk_level > 0.4:
        reward += -20 if features[0] > price_change_threshold else 0  # Moderate negative for bullish features

    # **Priority 2 — TREND FOLLOWING (when risk is low)**
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[2] > 0:  # Positive momentum alignment
            reward += 15
        elif trend_direction < 0 and features[2] < 0:  # Negative momentum alignment
            reward += 15

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < rsi_threshold_low:  # Oversold condition
            reward += 10  # Reward for buying in oversold conditions
        elif features[3] > rsi_threshold_high:  # Overbought condition
            reward += 10  # Reward for selling in overbought conditions

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds