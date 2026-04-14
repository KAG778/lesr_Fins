import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices

    features = []

    # Feature 1: Price Momentum (C[n] - C[n-21]) / C[n-21]
    price_momentum = (closing_prices[-1] - closing_prices[-21]) / closing_prices[-21] if len(closing_prices) > 21 and closing_prices[-21] != 0 else 0.0
    features.append(price_momentum)

    # Feature 2: Average Volume Change Rate (Current Volume vs. 20-day average)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0.0
    volume_change_rate = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0.0
    features.append(volume_change_rate)

    # Feature 3: 20-Day Volatility (Standard deviation of closing prices)
    volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(volatility)

    # Feature 4: Average True Range (ATR) for volatility estimation
    true_ranges = np.abs(np.diff(closing_prices)) if len(closing_prices) > 1 else np.array([0])
    atr = np.mean(true_ranges[-20:]) if len(true_ranges) >= 20 else 0.0
    features.append(atr)

    # Feature 5: Price Range (High - Low) over the last 20 days
    price_range = np.max(high_prices[-20:]) - np.min(low_prices[-20:]) if len(high_prices) >= 20 and len(low_prices) >= 20 else 0.0
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
        if features[0] < 0:  # Recent price momentum is negative, consider selling
            reward += 10.0  # Mild positive for SELL

    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 10.0 * features[0]  # Reward for momentum alignment based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition based on momentum
            reward += 5.0  # Reward for potential buy
        elif features[0] > 0.01:  # Overbought condition based on momentum
            reward += 5.0  # Reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))