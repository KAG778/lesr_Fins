import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: 10-Day Price Momentum (current price vs price 10 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0.0

    # Feature 2: 5-Day Average True Range (ATR) for volatility measurement
    high_prices = s[2::6][:10]  # Extract high prices for the last 10 days
    low_prices = s[3::6][:10]    # Extract low prices for the last 10 days
    close_prices = closing_prices[-10:]  # Last 10 closing prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - close_prices[:-1]), 
                               np.abs(low_prices[1:] - close_prices[:-1])))
    atr = np.mean(tr) if len(tr) > 0 else 0

    # Feature 3: Volume Change (current volume vs average volume of last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1.0  # Avoid division by zero
    current_volume = volumes[-1]
    volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0

    features = [price_momentum, atr, volume_change]
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
        reward -= 40.0  # Strong negative for BUY signals during high risk
        reward += 10.0 * features[2]  # Mild positive for SELL-aligned features (volume change)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Aligning reward with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Positive reward for potential buy
        elif features[0] > 0:  # Overbought condition
            reward -= 5.0  # Penalty for buying in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))