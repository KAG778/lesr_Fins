import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    high_prices = s[2::6]      # Extract high prices
    low_prices = s[3::6]       # Extract low prices

    features = []

    # Feature 1: 10-Day Price Momentum (current close vs close 10 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0.0
    features.append(price_momentum)

    # Feature 2: 10-Day Volatility (standard deviation of returns)
    if len(closing_prices) >= 10:
        returns = np.diff(closing_prices[-10:]) / closing_prices[-11:-1]
        volatility = np.std(returns)
    else:
        volatility = 0.0
    features.append(volatility)

    # Feature 3: 10-Day Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-10:]) if len(tr) >= 10 else 0.0
    features.append(atr)

    # Feature 4: Volume Change (current volume vs average volume of last 10 days)
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 1.0  # Avoid division by zero
    current_volume = volumes[-1]
    volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
    features.append(volume_change)

    # Feature 5: Price Range Percentage (high-low over the last 10 days)
    price_range_pct = (np.max(high_prices[-10:]) - np.min(low_prices[-10:])) / np.mean(low_prices[-10:]) if len(low_prices) >= 10 and np.mean(low_prices[-10:]) > 0 else 0.0
    features.append(price_range_pct)

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
        reward -= 40.0  # Strong negative for BUY signals
        reward += 10.0 * features[3]  # Mild positive for SELL-aligned features (volume change)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Reward for aligning with price momentum

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