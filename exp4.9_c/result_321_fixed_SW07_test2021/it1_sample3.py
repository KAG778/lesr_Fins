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
    
    avg_gain = np.mean(gains[-14:]) if len(gains[-14:]) > 0 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses[-14:]) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 3: Volume Change (Today vs Yesterday)
    volumes = s[4:120:6]  # Extract volumes
    volume_today = volumes[-1] if len(volumes) > 0 else 0
    volume_yesterday = volumes[-2] if len(volumes) > 1 else 0
    volume_change = (volume_today - volume_yesterday) / volume_yesterday if volume_yesterday > 0 else 0
    features.append(volume_change)

    # Feature 4: Average True Range (ATR) for volatility measurement
    highs = s[2:120:6]  # Extract high prices
    lows = s[3:120:6]   # Extract low prices
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closing_prices[:-1]), np.abs(lows[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) > 0 else 0
    features.append(atr)

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
        reward += 10.0 * (features[1] < 30)  # Mild positive for SELL if RSI is low (oversold)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Reward based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition
            reward += 15.0  # Strong positive for buying
        elif features[1] > 70:  # Overbought condition
            reward -= 10.0  # Strong negative for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))