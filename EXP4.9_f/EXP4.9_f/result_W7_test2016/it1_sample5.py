import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    features = []

    # Feature 1: Relative Price Change (last day compared to the average of previous days)
    recent_avg_price = np.mean(closing_prices[:-1]) if len(closing_prices) > 1 else closing_prices[0]
    relative_price_change = (closing_prices[-1] - recent_avg_price) / (recent_avg_price + 1e-10)  # Avoid division by zero
    features.append(relative_price_change)

    # Feature 2: Z-score of Volume (detect volume spikes)
    if len(volumes) > 1:
        volume_mean = np.mean(volumes[:-1])
        volume_std = np.std(volumes[:-1])
        z_score_volume = (volumes[-1] - volume_mean) / (volume_std + 1e-10)
    else:
        z_score_volume = 0
    features.append(z_score_volume)

    # Feature 3: Average True Range (ATR) over the last 14 days
    if len(closing_prices) >= 14:
        tr = np.maximum(closing_prices[-14:] - closing_prices[-15:], 
                        np.maximum(closing_prices[-14:] - closing_prices[-1], 
                                   closing_prices[-1] - closing_prices[-15:]))
        atr = np.mean(tr)
    else:
        atr = 0  # Not enough data for ATR
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    
    # Dynamic thresholds based on historical data
    avg_risk = 0.5  # This could be dynamically calculated based on historical data
    avg_volatility = 0.3  # This could be dynamically calculated based on historical data

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > avg_risk + 0.2:  # High-risk threshold
        reward -= 50 if features[0] > 0 else 0  # Strong negative for BUY-aligned features
        reward += 10 if features[1] < 0 else 0  # Mild positive for SELL-aligned features
    elif risk_level > avg_risk:
        reward -= 20 if features[0] > 0 else 0  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < avg_risk:  # Trend alignment with low risk
        if trend_direction > 0.3:
            reward += 20 if features[0] > 0 else 0  # Positive for upward alignment
        elif trend_direction < -0.3:
            reward += 20 if features[0] < 0 else 0  # Positive for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < avg_risk - 0.2:  # Sideways with low risk
        reward += 15 if features[0] < 0 else 0  # Reward mean-reversion features
        reward -= 5 if features[0] > 0 else 0  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > avg_volatility and risk_level < avg_risk:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure the reward is within the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward