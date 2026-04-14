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

    # Feature 4: Volatility (Standard deviation of closing prices over the last 20 days)
    volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0

    # Feature 5: RSI (Relative Strength Index over the last 14 days)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices)

    # Combine all features into a single array
    features = [price_change, average_volume, price_momentum, volatility, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]

    # Dynamic thresholds based on historical data
    avg_risk = 0.5  # Example average risk level based on historical data
    std_risk = 0.2  # Example standard deviation of risk level

    # Setting initial reward
    reward = 0.0

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > avg_risk + 1.5 * std_risk:  # Strong risk
        reward += -40 if features[0] > 0 else 5  # Buy aligned feature
    elif risk_level > avg_risk + 0.5 * std_risk:  # Moderate risk
        reward += -20 if features[0] > 0 else 0  # Buy aligned feature

    # **Priority 2 — TREND FOLLOWING**
    elif abs(trend_direction) > 0.3 and risk_level < avg_risk:
        if trend_direction > 0.3 and features[0] > 0:  # Buy aligned
            reward += 15
        elif trend_direction < -0.3 and features[0] < 0:  # Sell aligned
            reward += 15

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    elif abs(trend_direction) < 0.3 and risk_level < avg_risk - 0.5 * std_risk:
        if features[4] < 30:  # RSI oversold
            reward += 10  # Reward buying in oversold conditions
        elif features[4] > 70:  # RSI overbought
            reward += 10  # Reward selling in overbought conditions

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < avg_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds