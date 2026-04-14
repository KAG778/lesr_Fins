import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices

    # Feature 1: Mean and Standard Deviation of daily returns over the last 20 days
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.extend([mean_return, volatility])

    # Feature 2: Relative Strength Index (RSI) over the last 14 days
    def compute_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain[-period:]) > 0 else 0
        avg_loss = np.mean(loss[-period:]) if len(loss[-period:]) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = compute_rsi(closing_prices[-14:]) if len(closing_prices[-14:]) == 14 else 50
    features.append(rsi)

    # Feature 3: Price momentum (simple return over the last 5 days)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_risk_threshold = 0.5  # This could be dynamically calculated based on historical data
    high_risk_threshold = historical_risk_threshold + 0.2
    low_risk_threshold = historical_risk_threshold - 0.2

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -50  # Strong negative for BUY-aligned features
    elif risk_level > low_risk_threshold:
        reward += 20  # Mild positive for SELL signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:
            reward += 40  # Strong positive for upward momentum
        elif trend_direction < 0:
            reward += 40  # Strong positive for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        rsi = enhanced_s[123][2]  # Assuming RSI is the third feature
        if rsi < 30:
            reward += 30  # Buy signal when oversold
        elif rsi > 70:
            reward += -30  # Penalize for buying in an overbought market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward