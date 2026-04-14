import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    
    # Feature 1: Adaptive Moving Average (10-day vs 30-day)
    if len(closing_prices) >= 30:
        short_ma = np.mean(closing_prices[-10:])
        long_ma = np.mean(closing_prices[-30:])
        ma_diff = short_ma - long_ma
    else:
        ma_diff = 0

    # Feature 2: Volatility Measure (Historical Standard Deviation)
    historical_volatility = np.std(closing_prices[-10:]) if len(closing_prices[-10:]) > 0 else 0

    # Feature 3: Z-score of Volume (how current volume compares to historical average)
    avg_volume = np.mean(volumes[-10:]) if len(volumes[-10:]) > 0 else 1  # Avoid division by zero
    current_volume = volumes[0]  # Current volume
    z_score_volume = (current_volume - avg_volume) / historical_volatility if historical_volatility > 0 else 0

    features = [ma_diff, historical_volatility, z_score_volume]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds for risk management
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        reward += 10 * (1 - risk_level)  # Mild positive reward for SELL based on lower risk
    elif risk_level > risk_threshold_medium:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        else:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds