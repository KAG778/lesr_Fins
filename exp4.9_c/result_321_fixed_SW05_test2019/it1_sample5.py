import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Price Momentum (current price vs price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-5] if len(closing_prices) > 5 else 0.0

    # Feature 2: Volume Change (current volume vs average volume of last 5 days)
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[-5:]) if len(volumes) > 5 else 1.0  # Avoid division by zero
    volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0

    # Feature 3: Relative Strength Index (RSI) for mean reversion
    deltas = np.diff(closing_prices)
    gain = np.mean(deltas[deltas > 0]) if len(deltas) > 0 else 0
    loss = -np.mean(deltas[deltas < 0]) if len(deltas) > 0 else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [price_momentum, volume_change, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY signals
        # MILD POSITIVE for SELL-aligned features
        reward += 10.0 * features[1]  # Use volume change as a proxy for SELL
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Use price momentum for alignment with trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Oversold condition (RSI < 30)
            reward += 5.0  # Positive reward for potential buy
        elif features[2] > 70:  # Overbought condition (RSI > 70)
            reward -= 5.0  # Negative for buying in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))