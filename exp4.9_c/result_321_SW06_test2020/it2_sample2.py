import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volumes = s[4::6]  # Volumes

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
    
    # Feature 2: 14-day Exponential Moving Average of Daily Returns
    if len(daily_returns) > 14:
        ema_daily_return = np.mean(daily_returns[-14:])  # Simple EMA for simplicity
    else:
        ema_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
    features.append(ema_daily_return)

    # Feature 3: Recent Maximum Drawdown over the last 30 days
    if len(closing_prices) >= 30:
        peak = np.maximum.accumulate(closing_prices[-30:])
        drawdown = (peak - closing_prices[-30:]) / peak
        max_drawdown = np.max(drawdown)
    else:
        max_drawdown = 0  # Not enough data to calculate
    features.append(max_drawdown)

    # Feature 4: Momentum Indicator (Price Change / Volume Change)
    if len(closing_prices) > 1:
        price_change = closing_prices[-1] - closing_prices[-2]
        volume_change = volumes[-1] - volumes[-2]
        momentum = price_change / (volume_change + 1e-8)  # Avoid division by zero
    else:
        momentum = 0.0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]

    # Calculate relative thresholds based on historical data
    historical_risk = np.std(features[2])  # Using max drawdown as a risk measure
    historical_volatility = np.std(features[0])  # Using ATR as a volatility measure

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative reward for BUY-aligned features
        if features[3] < 0:  # If momentum is negative
            reward += 10  # Mild positive reward for SELL-aligned features
        return np.clip(reward, -100, 100)

    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[3] > 0:  # Uptrend and positive momentum
            reward += 20  # Align with bullish momentum
        elif trend_direction < 0 and features[3] < 0:  # Downtrend and negative momentum
            reward += 20  # Align with bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition (RSI)
            reward += 15  # Reward for buying
        elif features[1] > 70:  # Overbought condition (RSI)
            reward -= 15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clipping the reward to the range [-100, 100]
    return np.clip(reward, -100, 100)