import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Feature 1: 20-day moving average of closing prices
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    features.append(moving_avg)

    # Feature 2: 20-day volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(volatility)
    
    # Feature 3: Average True Range (ATR) over the last 14 days for volatility
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:], low_prices[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    # Feature 4: Rate of Change (ROC) for momentum (over 14 days)
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0
    features.append(roc)

    # Feature 5: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = -np.where(daily_returns < 0, daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Use historical data to calculate thresholds
    historical_risk_level = 0.5  # Placeholder for historical average risk
    historical_std_risk = 0.1  # Placeholder for historical standard deviation of risk

    high_risk_threshold = historical_risk_level + historical_std_risk
    low_risk_threshold = historical_risk_level - historical_std_risk

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for risky BUY-aligned features
        return np.clip(reward, -100, 100)  # Immediate return to prioritize risk management

    elif risk_level > low_risk_threshold:
        reward += 20  # Mild positive for SELL signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:  # Upward momentum
            reward += 30  # Positive reward for upward momentum
        elif trend_direction < 0:  # Downward momentum
            reward += 30  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        rsi = enhanced_s[123][4]  # Assuming RSI is the fifth feature
        if rsi < 30:
            reward += 30  # Buy signal when oversold
        elif rsi > 70:
            reward -= 30  # Strong negative for buying in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]