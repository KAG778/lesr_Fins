import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Momentum (Current closing price - closing price 5 days ago)
    price_momentum = s[114] - s[108]  # Close of day 19 - Close of day 18
    features.append(price_momentum)

    # Feature 2: Relative Strength Index (RSI)
    closing_prices = s[0:120:6]  # Extract closing prices
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

    avg_gain = np.mean(gains[-14:]) if len(gains[-14:]) > 0 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses[-14:]) > 0 else 0

    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 3: Volume Change (Current Volume - Volume 5 days ago)
    current_volume = s[114 + 4]  # Volume of day 19
    previous_volume = s[108 + 4]  # Volume of day 18
    volume_change = (current_volume - previous_volume) / previous_volume if previous_volume > 0 else 0
    features.append(volume_change)

    # Feature 4: Price Range Over the Last 20 Days
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    price_range = np.max(high_prices) - np.min(low_prices)  # Max high - Min low
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
        reward -= 40.0  # Strong negative for BUY
        reward += 5.0 * features[1]  # Mild positive for SELL if RSI is favorable
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Reward momentum alignment: positive if trend aligns with price momentum
        reward += trend_direction * features[0] * 10.0  # Price momentum reward

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition (RSI)
            reward += 15.0  # Strong buy signal
        elif features[1] > 70:  # Overbought condition (RSI)
            reward -= 10.0  # Strong sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))