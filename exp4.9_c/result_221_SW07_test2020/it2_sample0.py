import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Exponential Moving Average (EMA) for trend detection
    if len(closing_prices) >= 20:
        ema_short = np.mean(closing_prices[-10:])
        ema_long = np.mean(closing_prices[-20:])
        ema_trend = (ema_short - ema_long) / ema_long if ema_long != 0 else 0
    else:
        ema_trend = 0
    features.append(ema_trend)

    # Feature 2: Average True Range (ATR) for volatility measure
    high_prices = s[2::6]
    low_prices = s[3::6]
    if len(high_prices) >= 14:
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    else:
        atr = 0
    features.append(atr)

    # Feature 3: Z-score of Daily Returns to identify extreme conditions
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    if len(daily_returns) >= 20:
        mean_return = np.mean(daily_returns[-20:])
        std_return = np.std(daily_returns[-20:])
        z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    # Feature 4: Rate of Change (ROC) to identify momentum
    if len(closing_prices) >= 20:
        roc = (closing_prices[0] - closing_prices[19]) / closing_prices[19] if closing_prices[19] != 0 else 0
    else:
        roc = 0
    features.append(roc)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate historical standard deviations for dynamic thresholds
    historical_std = np.std(features)
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if features[0] > 0:  # Assuming positive trend is a BUY signal
            reward -= np.random.uniform(40, 60)  # Strong penalty
        if features[0] < 0:  # Assuming negative trend is a SELL signal
            reward += np.random.uniform(10, 20)  # Mild reward
    
    elif risk_level > low_risk_threshold:
        if features[0] > 0:  # Positive trend is a BUY signal
            reward -= np.random.uniform(10, 30)  # Moderate penalty

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend with positive momentum
            reward += np.random.uniform(10, 30)
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend with negative momentum
            reward += np.random.uniform(10, 30)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Oversold condition based on Z-score
            reward += np.random.uniform(10, 20)  # Reward for buying in oversold condition
        elif features[2] > 1:  # Overbought condition based on Z-score
            reward -= np.random.uniform(10, 20)  # Penalize buying in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds