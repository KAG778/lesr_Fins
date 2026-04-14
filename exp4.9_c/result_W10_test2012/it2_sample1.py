import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Exponential Moving Average (EMA) over the last 20 days
    ema = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    features.append(ema)

    # Feature 2: Average True Range (ATR) for volatility (14-day)
    high_prices = s[2::6]
    low_prices = s[3::6]
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:], low_prices[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    # Feature 3: Mean daily return over the last 20 days
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(mean_daily_return)

    # Feature 4: Rate of Change (ROC) for momentum (14-day)
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0
    features.append(roc)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate thresholds based on historical data
    historical_risk_level = np.std(enhanced_s[123])  # Historical risk based on feature volatility
    avg_risk_level = np.mean(enhanced_s[123])  # Average risk level from features

    # Priority 1 — RISK MANAGEMENT
    if risk_level > (avg_risk_level + 1.5 * historical_risk_level):  # Strong risk
        reward += -50  # Strong negative reward for risky BUY-aligned features
        return max(-100, reward)  # Immediate return to prioritize risk management
    elif risk_level > (avg_risk_level + 0.5 * historical_risk_level):  # Moderate risk
        reward += -20  # Moderate negative reward for BUY signals

    # Extract features for further evaluation
    features = enhanced_s[123:]

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < (avg_risk_level):  # Low risk condition
        if trend_direction > 0 and features[3] > 0:  # Upward momentum
            reward += 40  # Strong positive reward for upward alignment
        elif trend_direction < 0 and features[3] < 0:  # Downward momentum
            reward += 40  # Strong positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < (avg_risk_level):  # Low risk condition
        if features[2] < 30:  # RSI indicates oversold
            reward += 30  # Buy signal when oversold
        elif features[2] > 70:  # RSI indicates overbought
            reward += -30  # Strong negative for buying in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < (avg_risk_level):
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward