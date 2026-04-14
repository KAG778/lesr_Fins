import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    volumes = s[4:120:6]          # Extract volumes

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if len(closing_prices) >= 6 and closing_prices[5] != 0 else 0.0

    # Feature 2: Average Volume Change (over the last 5 days)
    volume_change = (volumes[0] - np.mean(volumes[1:6])) / np.mean(volumes[1:6]) if len(volumes) > 5 and np.mean(volumes[1:6]) != 0 else 0.0

    # Feature 3: Price Range (High - Low)
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 and len(low_prices) > 0 else 0.0

    # Feature 4: Relative Strength Index (RSI) calculation
    gain = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
    loss = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) > 14 else 0.0
    avg_loss = np.mean(loss[-14:]) if len(loss) > 14 else 0.0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))

    # Return the features as a numpy array
    features = [price_momentum, volume_change, price_range, rsi]
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
        reward += 10.0 if features[0] < 0 else 0  # Mild positive if price momentum is negative (suggest SELL)
    elif risk_level > 0.4:
        reward -= 10.0 if features[0] > 0 else 0  # Moderate negative for positive momentum (suggest BUY)

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Align reward with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI < 30 indicates oversold
            reward += 5.0  # Positive for potential buy signal
        elif features[3] > 70:  # RSI > 70 indicates overbought
            reward += 5.0  # Positive for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))