import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    highs = s[2:120:6]            # Extract high prices
    lows = s[3:120:6]             # Extract low prices

    # Feature 1: Price Momentum (current closing price vs moving average)
    moving_average = np.mean(closing_prices[-20:])  # 20-day moving average
    price_momentum = (closing_prices[-1] - moving_average) / (moving_average + 1e-10)

    # Feature 2: Volume Change Ratio (current volume vs moving average volume)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1e-10  # Avoid division by zero
    volume_change_ratio = (volumes[-1] - avg_volume) / (avg_volume + 1e-10)

    # Feature 3: Risk-Adjusted Price Range (normalized by the current price)
    price_range = (np.max(highs[-20:]) - np.min(lows[-20:])) / (closing_prices[-1] + 1e-10)

    return np.array([price_momentum, volume_change_ratio, price_range])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0
    price_momentum = features[0]
    volume_change_ratio = features[1]
    price_range = features[2]

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50.0 * (price_momentum if price_momentum > 0 else 1)  # Strong negative for BUY-aligned features
        reward += 5.0 * (volume_change_ratio < 0)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20.0 * price_momentum  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * price_momentum * 20.0  # Reward for following trend momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if price_momentum < -0.01:  # Oversold condition
            reward += 10.0  # Encourage BUY
        elif price_momentum > 0.01:  # Overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))