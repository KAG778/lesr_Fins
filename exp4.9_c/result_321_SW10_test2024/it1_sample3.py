import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    opening_prices = s[1:120:6]  # Extract opening prices
    volumes = s[4:120:6]          # Extract trading volumes
    highs = s[2:120:6]            # Extract high prices
    lows = s[3:120:6]             # Extract low prices

    # Feature 1: Price Change (current vs. previous)
    price_change = (closing_prices[-1] - closing_prices[-2]) / (closing_prices[-2] + 1e-10)

    # Feature 2: Average Volume Change over last 20 days
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0.0

    # Feature 3: Volatility (Standard Deviation of Price Changes)
    price_changes = np.diff(closing_prices)
    volatility = np.std(price_changes) if len(price_changes) > 0 else 0.0

    # Feature 4: Price Range (High - Low) normalized by closing price
    price_range = (highs[-1] - lows[-1]) / (closing_prices[-1] + 1e-10)

    return np.array([price_change, avg_volume, volatility, price_range])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0
    price_change = features[0]
    avg_volume = features[1]
    volatility = features[2]
    price_range = features[3]

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 5.0 * (1 - avg_volume)  # Mild positive for SELL-aligned features (less volume)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 10.0 * price_change * trend_direction  # Reward for following trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if price_change < -0.01:  # Oversold condition
            reward += 10.0  # Encourage BUY
        elif price_change > 0.01:  # Overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))