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

    # Feature 2: Standard Deviation of Returns (volatility) over the last 10 days
    if len(closing_prices) >= 10:
        returns = np.diff(closing_prices[-10:]) / closing_prices[-11:-1]
        volatility = np.std(returns)
    else:
        volatility = 0.0
    features.append(volatility)

    # Feature 3: Volume Change Percentage (current volume vs average volume over last 10 days)
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 1.0  # Prevent division by zero
    current_volume = volumes[-1] if len(volumes) > 0 else 0
    volume_change_pct = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
    features.append(volume_change_pct)

    # Feature 4: Price Range Percentage (high-low over the last 10 days)
    price_range_pct = (np.max(high_prices[-10:]) - np.min(low_prices[-10:])) / np.mean(low_prices[-10:]) if len(low_prices) >= 10 and np.mean(low_prices[-10:]) > 0 else 0.0
    features.append(price_range_pct)

    # Feature 5: Relative Strength Index (RSI) based on closing prices
    deltas = np.diff(closing_prices)
    gain = np.mean(deltas[deltas > 0]) if len(deltas) > 0 else 0
    loss = -np.mean(deltas[deltas < 0]) if len(deltas) > 0 else 0
    rs = gain / loss if loss > 0 else 0
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
        reward -= 40.0  # Strong negative for BUY signals
        reward += 10.0 * features[2]  # Mild positive for SELL-aligned features (volume change)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Reward for aligning with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[4] < 30:  # Oversold condition (RSI < 30)
            reward += 5.0  # Positive reward for potential buy
        elif features[4] > 70:  # Overbought condition (RSI > 70)
            reward -= 5.0  # Negative for buying in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))