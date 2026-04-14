import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (0, 6, 12, ..., 114)
    volumes = s[4::6]          # Extract volumes (4, 10, 16, ..., 114)

    # Feature 1: Price Change (%)
    price_change = 0
    if len(closing_prices) > 1 and closing_prices[-2] != 0:  # Check for division by zero
        price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Feature 2: Average Volume
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0

    # Feature 3: Price Momentum (e.g., difference from 5 days ago)
    price_momentum = 0
    if len(closing_prices) > 6:  # Check if we have enough data
        price_momentum = closing_prices[-1] - closing_prices[-6]  # Current - 5 days ago

    # Feature 4: Standard Deviation of Closing Prices (for volatility estimation)
    price_std = np.std(closing_prices) if len(closing_prices) > 1 else 0
    
    # Feature 5: Moving Average (20-day)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0

    # Feature 6: Price Range (high - low over the last 10 days)
    price_range = np.max(closing_prices[-10:]) - np.min(closing_prices[-10:]) if len(closing_prices) >= 10 else 0

    features = [price_change, average_volume, price_momentum, price_std, moving_average, price_range]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical thresholds based on the features
    price_change_threshold = np.std(features[0]) * 1.5
    volume_threshold = np.std(features[1]) * 1.5
    momentum_threshold = np.std(features[2]) * 1.5

    # **Priority 1 — RISK MANAGEMENT**
    if risk_level > 0.7:
        reward += -40 if features[0] > price_change_threshold else 5  # Strong negative for buying in high risk
    elif risk_level > 0.4:
        reward += -20 if features[0] > price_change_threshold else 0  # Moderate negative for buying in elevated risk

    # **Priority 2 — TREND FOLLOWING**
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[2] > momentum_threshold:  # Align with upward momentum
            reward += 15
        elif trend_direction < -0.3 and features[2] < -momentum_threshold:  # Align with downward momentum
            reward += 15

    # **Priority 3 — SIDEWAYS / MEAN REVERSION**
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -price_change_threshold:  # Oversold condition
            reward += 10
        elif features[0] > price_change_threshold:  # Overbought condition
            reward += 10

    # **Priority 4 — HIGH VOLATILITY**
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds