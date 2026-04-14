import numpy as np

def revise_state(s):
    # Extract relevant price and volume data from the raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract volumes

    # Feature 1: Price Momentum (percentage change from the previous close)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) > 1 and closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Volume Change (percentage change from 5 days ago)
    avg_volume_change = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) if len(volumes) > 5 and np.mean(volumes[-5:]) != 0 else 0.0
    
    # Feature 3: Average True Range (ATR) over the last 14 days
    true_ranges = []
    for i in range(1, len(closing_prices)):
        high = high_prices[i]
        low = low_prices[i]
        previous_close = closing_prices[i - 1]
        tr = max(high - low, abs(high - previous_close), abs(low - previous_close))
        true_ranges.append(tr)
    
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0

    # Feature 4: Stochastic Oscillator
    min_low = np.min(low_prices[-14:]) if len(low_prices) >= 14 else 0.0
    max_high = np.max(high_prices[-14:]) if len(high_prices) >= 14 else 0.0
    stochastic_oscillator = ((closing_prices[-1] - min_low) / (max_high - min_low)) * 100 if max_high > min_low else 0.0

    features = [price_momentum, avg_volume_change, atr, stochastic_oscillator]
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
        reward -= 10.0 if features[0] > 0 else 0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += features[0] * 10.0  # Align reward with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI < 30 indicates oversold
            reward += 5.0  # Positive for buying
        elif features[3] > 70:  # RSI > 70 indicates overbought
            reward += 5.0  # Positive for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))