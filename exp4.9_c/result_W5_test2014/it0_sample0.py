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
    average_volume = np.mean(volumes)

    # Feature 3: Price Momentum (e.g., difference from 5 days ago)
    price_momentum = 0
    if len(closing_prices) > 5:  # Check if we have enough data
        price_momentum = closing_prices[-1] - closing_prices[-6]  # Current - 5 days ago

    features = [price_change, average_volume, price_momentum]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if enhanced_s[123] > 0:  # Assuming features[0] is related to BUY
            return -40  # Example strong negative reward
        # Mild positive reward for SELL-aligned features
        return 7  # Example mild positive reward

    if risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if enhanced_s[123] > 0:  # Assuming features[0] is related to BUY
            reward -= 10  # Example moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # Assuming features[0] is BUY-aligned
            reward += 15  # Positive reward for correct bullish signal
        elif trend_direction < -0.3 and enhanced_s[123] < 0:  # Assuming features[0] is SELL-aligned
            reward += 15  # Positive reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Assuming features[0] is SELL-aligned
            reward += 10  # Positive reward for mean-reversion sell
        elif enhanced_s[123] > 0:  # Assuming features[0] is BUY-aligned
            reward += 10  # Positive reward for mean-reversion buy

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)