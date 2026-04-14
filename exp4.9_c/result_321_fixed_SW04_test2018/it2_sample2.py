import numpy as np

def revise_state(s):
    # Extract relevant price and volume data from the raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract volumes

    # Feature 1: Price Change Rate (percentage change from the last close)
    price_change_rate = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Change (percentage change from the previous day)
    avg_volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0.0

    # Feature 3: Price Range (High - Low over the last day)
    price_range = high_prices[-1] - low_prices[-1] if len(high_prices) > 0 and len(low_prices) > 0 else 0.0

    # Feature 4: Average True Range (ATR) over the last 14 days
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.abs(high_prices[1:] - closing_prices[:-1]),
                             np.abs(low_prices[1:] - closing_prices[:-1]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0

    features = [price_change_rate, avg_volume_change, price_range, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Relative thresholds based on historical data (example values)
    risk_threshold = 0.5  # Should be based on historical standard deviations
    trend_threshold = 0.3  # Example value for trend determination

    # Priority 1: Risk Management
    if risk_level > risk_threshold:
        reward -= 40.0  # Strong negative for BUY-aligned features
        if features[0] < 0:  # If price change is negative, reward for SELL
            reward += 10.0  # Mild positive for SELL
    elif risk_level > 0.4:
        reward -= 10.0 if features[0] > 0 else 0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        reward += features[0] * 10.0  # Align reward with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition for buying
            reward += 5.0  # Positive for buying
        elif features[0] > 0.01:  # Overbought condition for selling
            reward += 5.0  # Positive for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))