import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]
    
    # Feature 1: 5-day Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0
    features.append(price_momentum)
    
    # Feature 2: Standard Deviation of Closing Prices (last 20 days)
    if len(closing_prices) >= 20:
        price_std = np.std(closing_prices[-20:])
    else:
        price_std = 0.0
    features.append(price_std)
    
    # Feature 3: Bollinger Bands (20-day)
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
        upper_band = moving_average + (2 * price_std)
        lower_band = moving_average - (2 * price_std)
        # Calculate distance from bands
        band_distance = (closing_prices[-1] - upper_band) if closing_prices[-1] > upper_band else (closing_prices[-1] - lower_band)
    else:
        band_distance = 0.0
    features.append(band_distance)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds based on past risk and volatility
    historical_risk_threshold = 0.5  # Placeholder for historical mean risk level
    historical_volatility_threshold = 0.5  # Placeholder for historical mean volatility level
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
        reward += 10 * np.random.uniform(0, 1)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < historical_risk_threshold:
        if trend_direction > 0.3:
            reward += 20  # Reward bullish momentum
        elif trend_direction < -0.3:
            reward += 20  # Reward bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion signals
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]