import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    # Feature 1: 20-day Moving Average of Daily Returns
    if len(daily_returns) >= 20:
        ma_daily_returns = np.mean(daily_returns[-20:])
    else:
        ma_daily_returns = 0
    features.append(ma_daily_returns)

    # Feature 2: Maximum Drawdown Ratio over the last 30 days
    if len(closing_prices) >= 30:
        peak = np.maximum.accumulate(closing_prices[-30:])
        drawdown = (peak - closing_prices[-30:]) / peak
        max_drawdown_ratio = np.max(drawdown)
    else:
        max_drawdown_ratio = 0
    features.append(max_drawdown_ratio)

    # Feature 3: 14-day Average True Range (ATR)
    if len(closing_prices) > 14:
        high = s[1::6]  # High prices
        low = s[2::6]   # Low prices
        tr = np.maximum(high[1:] - low[1:], 
                        np.maximum(abs(high[1:] - closing_prices[:-1]), 
                                   abs(low[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])
    else:
        atr = 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    avg_daily_return = features[0]
    max_drawdown_ratio = features[1]
    atr = features[2]

    # Calculate historical thresholds for risk management
    historical_std_return = np.std(features[:2]) if len(features) >= 2 else 1e-8  # Avoid division by zero
    high_risk_threshold = 0.7 * historical_std_return
    low_risk_threshold = 0.4 * historical_std_return

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative for BUY-aligned features
        reward += -40 if avg_daily_return > 0 else 5  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)

    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        reward += -20 if avg_daily_return > 0 else 0

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= low_risk_threshold:
        if trend_direction > 0 and avg_daily_return > 0:
            reward += 20  # Reward for positive trend alignment
        elif trend_direction < 0 and avg_daily_return < 0:
            reward += 20  # Reward for negative trend alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if avg_daily_return < 0:  # Oversold condition
            reward += 15  # Reward for buying
        elif avg_daily_return > 0:  # Overbought condition
            reward -= 15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level <= low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)