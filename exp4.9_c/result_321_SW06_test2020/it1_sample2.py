import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Volumes
    
    # Feature 1: Average True Range (ATR) over the last 14 days
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
    
    # Feature 2: Momentum Indicator (Price Change / Volume Change)
    if len(closing_prices) > 1:
        price_change = closing_prices[-1] - closing_prices[-2]
        volume_change = volumes[-1] - volumes[-2]
        momentum = price_change / (volume_change + 1e-8)  # Avoid division by zero
    else:
        momentum = 0.0
        
    features.append(momentum)
    
    # Feature 3: 14-day Exponential Moving Average of Daily Returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    if len(daily_returns) > 14:
        ema_daily_return = np.mean(daily_returns[-14:])  # Simple EMA for simplicity
    else:
        ema_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
        
    features.append(ema_daily_return)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Determine dynamic risk thresholds based on historical volatility
    historical_volatility = np.std(features[0])  # Using ATR or similar for historical volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative for BUY-aligned features
        reward -= 40  # Strong negative for risky conditions
        if features[1] < 0:  # If momentum is negative
            reward += 5  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[1] > 0:  # Uptrend and positive momentum
            reward += 20  # Align with bullish momentum
        elif trend_direction < 0 and features[1] < 0:  # Downtrend and negative momentum
            reward += 20  # Align with bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 0:  # Oversold condition (momentum negative)
            reward += 15  # Reward for buying
        elif features[1] > 0:  # Overbought condition (momentum positive)
            reward -= 15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)