import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0
    ema_50 = np.mean(closing_prices[-50:]) if len(closing_prices) >= 50 else 0
    ema_trend = (ema_10 - ema_50) / ema_50 if ema_50 != 0 else 0  # EMA trend indicator
    
    # Feature 2: Average True Range (ATR) for volatility measure
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                         np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    
    # Feature 3: Z-score of recent returns to identify extreme conditions
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0  # 20-day mean
    std_return = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 1  # 20-day std, avoid div 0
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0  # Z-score
    
    features = [ema_trend, atr, z_score]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive trend is a BUY signal
            reward -= np.random.uniform(40, 60)  # Strong penalty
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # Assuming negative trend is a SELL signal
            reward += np.random.uniform(10, 20)  # Mild reward
    
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Assuming positive trend is a BUY signal
            reward -= np.random.uniform(10, 30)  # Moderate penalty
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Positive trend alignment
                reward += np.random.uniform(10, 30)  # Positive reward
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Negative trend alignment
                reward += np.random.uniform(10, 30)  # Positive reward
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Oversold condition based on Z-score
            reward += np.random.uniform(10, 20)  # Reward for buying in oversold condition
        elif features[2] > 1:  # Overbought condition based on Z-score
            reward += np.random.uniform(10, 20)  # Reward for selling in overbought condition
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds