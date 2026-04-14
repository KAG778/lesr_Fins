import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Price Momentum (Current closing price - closing price 5 days ago)
    closing_prices = s[0:120:6]  # Extract closing prices
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(price_momentum)

    # Feature 2: Relative Strength Index (RSI)
    gains = []
    losses = []
    
    for i in range(1, len(closing_prices)):
        change = closing_prices[i] - closing_prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            losses.append(-change)
            gains.append(0)

    avg_gain = np.mean(gains[-14:]) if len(gains) > 0 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 3: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) > 0 else 0
    features.append(atr)

    # Feature 4: Volume Change (Current volume - average volume of the last 5 days)
    volumes = s[4:120:6]  # Extract volumes
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    current_volume = volumes[-1]  # Today's volume
    volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
    features.append(volume_change)

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
        reward -= 40.0  # Strong negative for BUY
        reward += 10.0 * (100 - features[1]) / 100  # Mild positive for SELL if RSI is low (oversold)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Reward based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition (RSI < 30)
            reward += 15.0  # Strong buy signal
        elif features[1] > 70:  # Overbought condition (RSI > 70)
            reward -= 10.0  # Strong sell signal

    # Priority 4: High Volatility
    if volatility_level > np.percentile(volatility_level, 75):  # High volatility threshold
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))