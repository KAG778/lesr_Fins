import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    
    features = []

    # Feature 1: Price Change Percentage (C[n] - C[n-1]) / C[n-1]
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_change_pct)

    # Feature 2: Average Volume over the last 20 days
    if len(volumes) >= 20:
        average_volume = np.mean(volumes[-20:])
    else:
        average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    features.append(average_volume)

    # Feature 3: Price Range (High - Low) over the last 20 days
    if len(high_prices) >= 20 and len(low_prices) >= 20:
        price_range = np.max(high_prices[-20:]) - np.min(low_prices[-20:])
    else:
        price_range = 0.0
    features.append(price_range)

    # Feature 4: 20-Day Volatility (standard deviation of price changes)
    price_changes = np.diff(closing_prices[-20:]) if len(closing_prices) >= 20 else np.array([0])
    volatility = np.std(price_changes) if len(price_changes) > 1 else 0.0
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
        reward += 10.0 if features[0] < 0 else 0  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 10.0 * features[0]  # Reward based on price change percentage

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Use historical volatility to define overbought/oversold thresholds
        oversold_threshold = -np.std(features[0:1]) * 0.5  # Use half of the standard deviation as threshold
        overbought_threshold = np.std(features[0:1]) * 0.5  # Use half of the standard deviation as threshold
        if features[0] < oversold_threshold:  # Oversold condition
            reward += 5.0  # Reward for potential buy
        elif features[0] > overbought_threshold:  # Overbought condition
            reward += 5.0  # Reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))