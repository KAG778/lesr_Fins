import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price Momentum (current close - 5 days ago close)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0.0

    # Feature 2: Volume Change (current volume vs. average volume of last 5 days)
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1.0  # Avoid division by zero
    volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0

    # Feature 3: Price Range Percentage (high-low over the last 20 days)
    high_prices = s[2::6]
    low_prices = s[3::6]
    price_range_pct = (np.max(high_prices[-20:]) - np.min(low_prices[-20:])) / np.mean(low_prices[-20:]) if len(low_prices) >= 20 and np.mean(low_prices[-20:]) > 0 else 0.0

    # Feature 4: Exponential Moving Average (EMA) of closing prices (last 5 days)
    alpha = 2 / (5 + 1)  # EMA smoothing factor for 5 days
    ema = closing_prices[-5:].copy() if len(closing_prices) >= 5 else closing_prices
    for i in range(1, len(ema)):
        ema[i] = alpha * closing_prices[-(i+1)] + (1 - alpha) * ema[i-1]

    features = [price_momentum, volume_change, price_range_pct, ema[-1]]
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
        reward += 5.0 * features[1]  # MILD POSITIVE for SELL-aligned features based on volume change
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive momentum
            reward += trend_direction * features[0] * 10.0  # Reward for aligning with trend
        elif features[0] < 0:  # Negative momentum
            reward += trend_direction * features[0] * 10.0  # Reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Oversold condition
            reward += 5.0  # Positive reward for potential buy
        elif features[0] > 0.05:  # Overbought condition
            reward -= 5.0  # Penalty for buying in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))