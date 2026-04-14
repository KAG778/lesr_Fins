import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    
    features = []

    # Feature 1: Price Change Percentage (C[n] - C[n-1]) / C[n-1]
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    
    # Feature 2: Average Volume over the last 20 days
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0

    # Feature 3: Average True Range (ATR) for volatility estimation
    price_diffs = np.abs(np.diff(closing_prices))
    atr = np.mean(price_diffs[-14:]) if len(price_diffs) >= 14 else 0  # 14-day average

    # Feature 4: Price Range (High - Low) over the last 20 days
    price_range = np.max(high_prices[-20:]) - np.min(low_prices[-20:]) if len(high_prices) >= 20 and len(low_prices) >= 20 else 0

    # Feature 5: Volume Change (Percentage Change from Previous Day)
    volume_change_pct = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0.0

    features.extend([price_change_pct, average_volume, atr, price_range, volume_change_pct])
    
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
        reward += 10.0 * features[0]  # Reward for momentum alignment based on price change percentage

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Use historical average and std to define overbought/oversold conditions
        mean_price_change = np.mean(features[0]) if features[0] != 0 else 1
        if features[0] < -mean_price_change * 0.01:  # Oversold condition
            reward += 5.0  # Reward for potential buy
        elif features[0] > mean_price_change * 0.01:  # Overbought condition
            reward += 5.0  # Reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))