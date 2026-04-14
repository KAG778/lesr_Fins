import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    # Feature 1: 20-day Average Daily Return
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(avg_daily_return)

    # Feature 2: Recent Volatility (Standard deviation of daily returns over the last 20 days)
    recent_volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(recent_volatility)

    # Feature 3: 14-day Relative Strength Index (RSI)
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / (avg_loss + 1e-8)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 4: 14-day Exponential Moving Average of Daily Returns
    if len(daily_returns) > 14:
        ema_daily_return = np.mean(daily_returns[-14:])  # Simple EMA for simplicity
    else:
        ema_daily_return = avg_daily_return  # Fallback to average daily return
    features.append(ema_daily_return)

    # Feature 5: Momentum Indicator (Price Change over last 5 days)
    if len(closing_prices) >= 6:
        momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]  # Price change percentage
    else:
        momentum = 0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    avg_daily_return = features[0]
    recent_volatility = features[1]
    rsi = features[2]
    ema_daily_return = features[3]
    momentum = features[4]
    
    # Calculate relative thresholds based on historical data
    historical_volatility = np.std(features[1]) if np.std(features[1]) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_medium = 0.4 * historical_volatility

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        reward += -40 if avg_daily_return > 0 else 10  # Mild positive for SELL
        return np.clip(reward, -100, 100)

    elif risk_level > risk_threshold_medium:
        # Moderate negative reward for BUY signals
        if avg_daily_return > 0:
            reward += -20

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0 and momentum > 0:  # Bullish trend with positive momentum
            reward += 20
        elif trend_direction < 0 and momentum < 0:  # Bearish trend with negative momentum
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward += 15
        elif rsi > 70:  # Overbought condition
            reward -= 15

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)