import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]
    opening_prices = s[1:120:6]
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    volumes = s[4:120:6]

    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - opening_prices[-1]) / opening_prices[-1] if opening_prices[-1] != 0 else 0.0

    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Price Range (high - low) of the last 20 days
    price_range = np.max(high_prices) - np.min(low_prices)

    # Feature 4: Relative Strength Index (RSI) calculation (simplified)
    gain = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
    loss = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) > 14 else 0.0
    avg_loss = np.mean(loss[-14:]) if len(loss) > 14 else 0.0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))

    # Return the features as a numpy array
    features = [price_change_pct, avg_volume, price_range, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= 40.0
        # Mild positive reward for SELL-aligned features
        reward += 5.0 if features[0] < 0 else 0  # negative price change aligns with SELL
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10.0 if features[0] > 0 else 0  # positive price change aligns with BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            # Positive reward for upward trend and upward price movement
            reward += features[0] * 10.0  # price change percentage
        else:
            # Positive reward for downward trend and downward price movement
            reward += -features[0] * 10.0  # negative of price change percentage

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI < 30 indicates oversold
            reward += 5.0  # mild positive for potential buy signal
        elif features[3] > 70:  # RSI > 70 indicates overbought
            reward += 5.0  # mild positive for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # reduce reward magnitude

    return float(np.clip(reward, -100, 100))