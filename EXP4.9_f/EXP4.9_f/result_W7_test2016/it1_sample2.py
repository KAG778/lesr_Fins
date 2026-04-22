import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # closing prices
    volumes = s[4::6]          # trading volumes
    
    # Feature 1: Rate of Change (RoC) - measures momentum
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) >= 6 else 0
    features.append(roc)

    # Feature 2: Volume Weighted Average Price (VWAP) - helps in trend analysis
    vwap = np.sum(closing_prices[-5:] * volumes[-5:]) / np.sum(volumes[-5:]) if np.sum(volumes[-5:]) > 0 else 0
    features.append(vwap)

    # Feature 3: Bollinger Bands - measures volatility and potential price reversals
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        features.append(upper_band)
        features.append(lower_band)
    else:
        features.append(0)  # Not enough data to calculate
        features.append(0)  # Not enough data to calculate

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]

    # Calculate thresholds based on historical data (this can be adapted to more sophisticated methods)
    avg_risk = np.mean([0.2, 0.4, 0.6, 0.8])  # Example historical risk levels
    std_risk = np.std([0.2, 0.4, 0.6, 0.8])
    high_risk_threshold = avg_risk + 1.5 * std_risk  # Adjusted threshold for high risk

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50 if features[0] > 0 else 0  # Strong negative for BUY features
        reward += 10 if features[0] < 0 else 0  # Mild positive for SELL features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20 if features[0] > 0 else 0  # Positive reward for upward momentum
        elif trend_direction < -0.3:
            reward += 20 if features[0] < 0 else 0  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15 if features[0] < 0 else 0  # Reward for buying in oversold conditions
        reward -= 15 if features[0] > 0 else 0  # Penalty for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Normalize reward to the range [-100, 100]
    return np.clip(reward, -100, 100)