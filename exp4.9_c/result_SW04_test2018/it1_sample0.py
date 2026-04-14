import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices of the last 20 days
    high_prices = s[2::6]      # High prices of the last 20 days
    low_prices = s[3::6]       # Low prices of the last 20 days
    volumes = s[4::6]          # Trading volumes of the last 20 days
    
    # Feature 1: Average True Range (ATR) - captures volatility
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 2: Exponential Moving Average (EMA) - captures trends
    ema = np.mean(closing_prices[-14:])  # 14-day EMA as a simple approximation

    # Feature 3: Volume Spike - compares current volume to average volume over last 5 days
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 1  # Avoid division by zero
    volume_spike = (volumes[-1] - avg_volume) / avg_volume  # Normalized volume change

    features = [atr, ema, volume_spike]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical volatility
    historical_volatility = np.std(enhanced_s[123:126])  # Use features for volatility context
    high_risk_threshold = 0.5 * historical_volatility
    low_risk_threshold = 0.25 * historical_volatility

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Strong penalty for risky buy signals
        reward += np.random.uniform(5, 10)    # Mild positive for sell signals
    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(5, 15)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for bullish features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        # Reward mean-reversion features (assumed features are in enhanced_s[123:])
        reward += 10  # Example positive reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50% during high volatility

    # Constrain reward within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return float(reward)