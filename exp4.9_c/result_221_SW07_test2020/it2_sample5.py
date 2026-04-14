import numpy as np

def revise_state(s):
    features = []
    
    # Closing prices for the last 20 days
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Exponential Moving Average (EMA) Momentum
    if len(closing_prices) >= 20:
        ema_short = np.mean(closing_prices[-10:])  # Short-term EMA
        ema_long = np.mean(closing_prices[-20:])  # Long-term EMA
        momentum = (ema_short - ema_long) / ema_long if ema_long != 0 else 0
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

    # Feature 3: Z-score of recent returns to identify mean-reversion opportunities
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    if len(daily_returns) >= 20:
        mean_return = np.mean(daily_returns[-20:])
        std_return = np.std(daily_returns[-20:])
        z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    else:
        z_score = 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(features)
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if features[0] > 0:  # Positive momentum implies a BUY signal
            reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        if features[0] < 0:  # Negative momentum implies a SELL signal
            reward += np.random.uniform(5, 10)    # Mild positive reward for SELL
    
    elif risk_level > low_risk_threshold:
        if features[0] > 0:  # Positive momentum implies a BUY signal
            reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Positive momentum
                reward += np.random.uniform(10, 20)  # Reward for alignment
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Negative momentum
                reward += np.random.uniform(10, 20)  # Reward for alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Oversold condition (Z-score)
            reward += np.random.uniform(10, 20)  # Reward for buying
        elif features[2] > 1:  # Overbought condition (Z-score)
            reward -= np.random.uniform(10, 20)  # Penalty for buying

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds