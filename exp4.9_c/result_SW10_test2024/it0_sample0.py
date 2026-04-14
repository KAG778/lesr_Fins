import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract required OHLCV data
    closing_prices = s[0:120:6]  # Get all closing prices
    volumes = s[4:120:6]          # Get all volumes

    # Feature 1: Price Change Percentage
    price_change_pct = np.zeros(19)
    for i in range(1, 20):
        if closing_prices[i-1] != 0:
            price_change_pct[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
        else:
            price_change_pct[i-1] = 0  # Handle division by zero

    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(volumes)

    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = -np.where(delta < 0, delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) > 14 else np.mean(gain)
    avg_loss = np.mean(loss[-14:]) if len(loss) > 14 else np.mean(loss)

    # Avoid division by zero
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Return only new features
    features = [price_change_pct[-1], avg_volume, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += -20  # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Assuming features[0] is a positive trend feature
        if trend_direction > 0 and enhanced_state[123] > 0:  # Assuming the feature indicates upward momentum
            reward += 20  # Positive reward for BUY
        elif trend_direction < 0 and enhanced_state[123] < 0:  # Assuming the feature indicates downward momentum
            reward += 20  # Positive reward for SELL

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming features[0] is an oversold signal for buy
        if enhanced_state[123] < 30:  # Example condition for oversold
            reward += 15  # Reward for potential buy
        # Assuming features[0] is an overbought signal for sell
        elif enhanced_state[123] > 70:  # Example condition for overbought
            reward += 15  # Reward for potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward