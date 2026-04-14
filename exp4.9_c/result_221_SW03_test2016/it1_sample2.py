import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Volatility-adjusted Momentum
    if len(closing_prices) >= 6:
        recent_momentum = closing_prices[-1] - closing_prices[-2]
        recent_volatility = np.std(closing_prices[-5:]) if np.std(closing_prices[-5:]) > 0 else 1e-10  # prevent div by zero
        vol_adjusted_momentum = recent_momentum / recent_volatility
    else:
        vol_adjusted_momentum = 0.0

    # Feature 2: Mean Reversion Z-Score
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        if std_price > 0:
            z_score = (closing_prices[-1] - mean_price) / std_price
        else:
            z_score = 0.0
    else:
        z_score = 0.0

    # Feature 3: Volume Spike Ratio
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1e-10  # prevent div by zero
    current_volume = volumes[-1]
    volume_spike_ratio = current_volume / avg_volume

    features = [vol_adjusted_momentum, z_score, volume_spike_ratio]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk assessment
    # Using historical std for relative thresholds (assuming we have access to historical data)
    historical_std_risk = 0.5  # Placeholder value, should be calculated from historical data
    historical_std_trend = 0.3  # Placeholder value
    historical_std_vol = 0.5  # Placeholder value

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_std_risk:
        reward -= 40  # Strong negative reward for BUY-aligned features
        reward += 7    # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > historical_std_trend and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        z_score = enhanced_s[123]  # Assuming the Z-Score is the first feature
        if z_score < -1:  # Oversold condition
            reward += 10  # Reward potential buy
        elif z_score > 1:  # Overbought condition
            reward += 10  # Reward potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std_vol and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds