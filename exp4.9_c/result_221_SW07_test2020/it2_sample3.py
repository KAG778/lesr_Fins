import numpy as np

def revise_state(s):
    features = []
    
    # Closing prices for the last 20 days
    closing_prices = s[0::6]
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) Momentum
    if len(closing_prices) >= 20:
        ema_10 = np.mean(closing_prices[-10:])
        ema_20 = np.mean(closing_prices[-20:])
        momentum = (ema_10 - ema_20) / ema_20 if ema_20 != 0 else 0
    else:
        momentum = 0
    features.append(momentum)

    # Feature 2: Average True Range (ATR) for volatility measure
    if len(closing_prices) >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    else:
        atr = 0
    features.append(atr)

    # Feature 3: Z-score of Recent Returns (Mean Reversion Indicator)
    if len(closing_prices) >= 20:
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]
        mean_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
        std_return = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 1
        z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    # Feature 4: Historical Volatility (last 20 days)
    if len(daily_returns) >= 20:
        historical_volatility = np.std(daily_returns[-20:])
    else:
        historical_volatility = 0
    features.append(historical_volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate thresholds based on historical standard deviation
    historical_std = np.std(features)
    if historical_std == 0:
        relative_threshold_high = 0.3  # Default threshold
        relative_threshold_low = -0.3
    else:
        relative_threshold_high = historical_std * 0.3
        relative_threshold_low = -historical_std * 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive momentum
            reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        if features[0] < 0:  # Negative momentum
            reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features

    elif risk_level > 0.4:
        if features[0] > 0:  # Positive momentum
            reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > relative_threshold_high and features[0] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(10, 20)
        elif trend_direction < relative_threshold_low and features[0] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Oversold condition (z-score)
            reward += np.random.uniform(10, 20)
        elif features[2] > 1:  # Overbought condition (z-score)
            reward -= np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds