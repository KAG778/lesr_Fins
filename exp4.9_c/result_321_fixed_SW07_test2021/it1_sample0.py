import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]         # Volumes

    # Feature 1: Price Momentum (Current closing price - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(price_momentum)

    # Feature 2: Average Volume Change (percentage)
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    volume_today = volumes[-1]
    volume_change = (volume_today - avg_volume) / avg_volume if avg_volume > 0 else 0
    features.append(volume_change)

    # Feature 3: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # High prices
    low_prices = s[3:120:6]   # Low prices
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:])
    true_ranges = np.maximum(true_ranges, closing_prices[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0
    features.append(atr)

    # Feature 4: Relative Strength Index (RSI) to gauge overbought/oversold conditions
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
        reward += 10.0 * (100 - features[3]) / 100  # Mild positive for SELL if RSI is in safe range
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Reward momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # Oversold condition (RSI < 30)
            reward += 15.0  # Strong buy signal in mean-reversion
        elif features[3] > 70:  # Overbought condition (RSI > 70)
            reward -= 10.0  # Strong sell signal in mean-reversion

    # Priority 4: High Volatility
    if volatility_level > np.percentile(volatility_level, 75):  # High volatility threshold
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))