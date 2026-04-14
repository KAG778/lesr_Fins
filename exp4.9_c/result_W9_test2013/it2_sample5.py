import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Price Change Percentage over the last 10 days
    if len(closing_prices) >= 10:
        price_change_pct = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if closing_prices[-10] != 0 else 0
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # Feature 2: Average Volume Change Percentage over the last 10 days
    if len(volumes) >= 10:
        avg_volume = np.mean(volumes[-10:])
        avg_volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    else:
        avg_volume_change = 0
    features.append(avg_volume_change)

    # Feature 3: Bollinger Bands (Lower Band) based on last 20 days
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        lower_band = rolling_mean - (rolling_std * 2)  # 2 standard deviations
    else:
        lower_band = closing_prices[-1] if len(closing_prices) > 0 else 0
    features.append(lower_band)

    # Feature 4: Price Momentum (current - previous day)
    if len(closing_prices) >= 2:
        price_momentum = closing_prices[-1] - closing_prices[-2]
    else:
        price_momentum = 0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract newly calculated features
    reward = 0.0

    # Calculate historical thresholds based on features
    historical_std = np.std(features)
    historical_mean = np.mean(features)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming feature 0 indicates bullish signal
            reward -= 50 * historical_std  # Strong negative for BUY
        else:
            reward += 10 * historical_std  # Mild positive for SELL
    elif risk_level > 0.4:
        if features[0] > 0:
            reward -= 25 * historical_std  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Aligning with an upward trend
            reward += 30 * historical_std  # Strong positive for bullish signals
        elif trend_direction < -0.3 and features[0] < 0:  # Aligning with a downward trend
            reward += 30 * historical_std  # Strong positive for bearish signals

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition
            reward += 15  # Reward for buying in oversold conditions
        elif features[0] > 0.1:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward