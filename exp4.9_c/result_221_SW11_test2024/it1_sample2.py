import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    features = []

    # Feature 1: Exponential Moving Average (EMA) over the last 10 days
    if len(closing_prices) >= 10:
        ema_10 = np.mean(closing_prices[-10:])  # Simple EMA approximation
    else:
        ema_10 = closing_prices[-1] if len(closing_prices) > 0 else 0
    features.append(ema_10)

    # Feature 2: Price Change Percentage over the last 5 days
    price_change_5d = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) >= 6 and closing_prices[-6] != 0 else 0
    features.append(price_change_5d)

    # Feature 3: Average True Range (ATR) over the last 14 days (volatility measure)
    if len(closing_prices) >= 15:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                   abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # Average of the True Range over the last 14 days
    else:
        atr = 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds for relative measures
    historical_std = np.std(enhanced_s[123:])  # Assuming features are at enhanced_s[123:]
    threshold_risk_high = 0.7 * historical_std
    threshold_risk_medium = 0.4 * historical_std
    threshold_trend_high = 0.3 * historical_std

    reward = 0.0

    # Priority 1: RISK MANAGEMENT
    if risk_level > threshold_risk_high:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features
        return np.clip(reward, -100, 100)

    if risk_level > threshold_risk_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > threshold_trend_high and risk_level < threshold_risk_medium:
        if trend_direction > threshold_trend_high:
            reward += np.random.uniform(10, 30)  # Positive reward for upward features
        elif trend_direction < -threshold_trend_high:
            reward += np.random.uniform(10, 30)  # Positive reward for downward features

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= threshold_trend_high and risk_level < threshold_risk_high:
        reward += np.random.uniform(-10, 10)  # Reward for mean-reversion features
        reward -= np.random.uniform(5, 15)     # Penalize breakout-chasing features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < threshold_risk_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds