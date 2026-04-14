import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volumes = s[4::6]  # Volumes
    
    # Feature 1: Recent Volatility (Standard deviation of daily returns over the last 20 days)
    if len(daily_returns) >= 20:
        recent_volatility = np.std(daily_returns[-20:])
    else:
        recent_volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
    features.append(recent_volatility)
    
    # Feature 2: Average Daily Return over the last 20 days
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    features.append(avg_daily_return)

    # Feature 3: Maximum Drawdown over the last 30 days
    if len(closing_prices) >= 30:
        peak = np.maximum.accumulate(closing_prices[-30:])
        drawdown = (peak - closing_prices[-30:]) / peak
        max_drawdown = np.max(drawdown)
    else:
        max_drawdown = 0  # Not enough data to calculate
    features.append(max_drawdown)

    # Feature 4: Average True Range (ATR) over the last 14 days
    if len(closing_prices) > 14:
        high = s[1::6]  # High prices
        low = s[2::6]   # Low prices
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(abs(high[1:] - closing_prices[:-1]), 
                                   abs(low[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # Average of the True Range
    else:
        atr = 0.0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Calculate thresholds based on historical data
    recent_volatility = features[0]  # Recent volatility
    avg_daily_return = features[1]    # Average daily return
    max_drawdown = features[2]         # Maximum drawdown
    atr = features[3]                   # Average True Range

    # Determine relative thresholds for risk management
    high_risk_threshold = 0.7 * np.mean([recent_volatility, np.std([recent_volatility, avg_daily_return])])
    medium_risk_threshold = 0.4 * np.mean([recent_volatility, np.std([recent_volatility, avg_daily_return])])

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward += -40 if avg_daily_return > 0 else 10  # Mild positive reward for SELL-aligned features
        return np.clip(reward, -100, 100)

    elif risk_level > medium_risk_threshold:
        # Moderate negative reward for BUY signals
        reward += -20 if avg_daily_return > 0 else 0

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0 and avg_daily_return > 0:  # Bullish trend and positive return
            reward += 20
        elif trend_direction < 0 and avg_daily_return < 0:  # Bearish trend and negative return
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if avg_daily_return < 0:  # Oversold condition (momentum negative)
            reward += 15  # Reward for buying
        elif avg_daily_return > 0:  # Overbought condition (momentum positive)
            reward -= 15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)