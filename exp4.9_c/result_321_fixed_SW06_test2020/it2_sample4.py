import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    
    features = []

    # Feature 1: 20-Day Price Momentum (Current price - Price 20 days ago)
    if len(closing_prices) > 20:
        price_momentum = closing_prices[-1] - closing_prices[-21]
    else:
        price_momentum = 0.0
    features.append(price_momentum)

    # Feature 2: Average Volume Change Rate over the last 20 days
    if len(volumes) > 1:
        volume_change_rate = np.mean(np.diff(volumes[-20:])) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else 0.0
    else:
        volume_change_rate = 0.0
    features.append(volume_change_rate)

    # Feature 3: 20-Day Volatility (Standard deviation of price changes)
    price_changes = np.diff(closing_prices[-20:]) if len(closing_prices) >= 20 else np.array([])
    volatility = np.std(price_changes) if len(price_changes) > 0 else 0.0
    features.append(volatility)

    # Feature 4: 20-Day Average True Range (ATR)
    true_ranges = np.maximum(0, high_prices[1:]-low_prices[1:])  # Simplified ATR calculation
    atr = np.mean(true_ranges[-20:]) if len(true_ranges) >= 20 else 0.0
    features.append(atr)

    # Feature 5: Price Range over the last 20 days (High - Low)
    if len(high_prices) > 20 and len(low_prices) > 20:
        price_range = np.max(high_prices[-20:]) - np.min(low_prices[-20:])
    else:
        price_range = 0.0
    features.append(price_range)

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
        # Mild positive for SELL-aligned features if price momentum is negative
        if features[0] < 0:  # Recent price momentum is negative, consider selling
            reward += 10.0
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 10.0 * features[0]  # Reward for momentum alignment based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition based on price momentum
            reward += 5.0  # Reward for potential buy
        elif features[0] > 0.01:  # Overbought condition based on price momentum
            reward += 5.0  # Reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    # Return reward value clipped to the range [-100, 100]
    return float(np.clip(reward, -100, 100))