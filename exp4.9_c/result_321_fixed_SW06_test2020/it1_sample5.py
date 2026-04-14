import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    price_changes = np.diff(closing_prices)  # Price changes for momentum calculation

    features = []

    # Feature 1: Price Momentum (last price change)
    price_momentum = price_changes[-1] if len(price_changes) > 0 else 0.0
    features.append(price_momentum)

    # Feature 2: Average Volume over the last 20 days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    features.append(average_volume)

    # Feature 3: Price Range (High - Low) over the last 20 days
    price_range = np.max(closing_prices) - np.min(closing_prices)
    features.append(price_range)

    # Feature 4: 20-Day Volatility (standard deviation of price changes)
    volatility = np.std(price_changes[-20:]) if len(price_changes) >= 20 else 0.0
    features.append(volatility)

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
        reward -= 40.0  # Strong negative for BUY-aligned features
        # Mild positive for SELL-aligned features based on recent price momentum
        if features[0] < 0:  # Recent price momentum is negative, consider selling
            reward += 5.0
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price momentum
            reward += 10.0 * features[0]  # Reward for alignment with trend
        elif features[0] < 0:  # Negative price momentum
            reward += 10.0 * -features[0]  # Penalize for contrarian bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 5.0  # Reward for potential buy
        elif features[0] > 0.01:  # Overbought condition
            reward += 5.0  # Reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))