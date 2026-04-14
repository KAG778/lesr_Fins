import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    # Feature 1: Average Daily Return over the last 19 days
    avg_daily_return = np.mean(daily_returns[-19:]) if len(daily_returns) >= 19 else 0
    features.append(avg_daily_return)

    # Feature 2: Volatility (Standard deviation of daily returns)
    volatility = np.std(daily_returns[-19:]) if len(daily_returns) >= 19 else 0
    features.append(volatility)

    # Feature 3: 14-day Relative Strength Index (RSI)
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / (avg_loss + 1e-8)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 4: 5-day Moving Average Convergence Divergence (MACD)
    short_ma = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    long_ma = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    macd = short_ma - long_ma
    features.append(macd)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    avg_daily_return = features[0]
    volatility = features[1]
    rsi = features[2]
    macd = features[3]
    
    # Determine thresholds based on historical volatility
    risk_threshold_high = np.mean(volatility) + 2 * np.std(volatility)  # Example threshold for high-risk
    risk_threshold_low = np.mean(volatility) + 1 * np.std(volatility)   # Example threshold for moderate-risk

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative for BUY
        if avg_daily_return < 0:  # Mild positive for SELL
            reward += 10
        return np.clip(reward, -100, 100)

    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and avg_daily_return > 0:  # Bullish trend and positive return
            reward += 20
        elif trend_direction < 0 and avg_daily_return < 0:  # Bearish trend and negative return
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward += 15
        elif rsi > 70:  # Overbought condition
            reward -= 15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clipping the reward to the range [-100, 100]
    return np.clip(reward, -100, 100)