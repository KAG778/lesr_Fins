import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]  # Every 6th element starting from index 0 is the closing price
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Rate of Change (Price Momentum)
    if len(closing_prices) > 1 and closing_prices[-2] != 0:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        momentum = 0.0
    features.append(momentum)

    # Feature 2: Average Volume Change (relative to historical average)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes) if len(volumes) > 0 else 0
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_change)

    # Feature 3: Historical Volatility (20-day)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns) * np.sqrt(20)  # Annualized volatility
    features.append(historical_volatility)

    # Feature 4: Bollinger Bands % (relative position)
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        if std_dev > 0:
            bb_percent = (closing_prices[-1] - (mean_price - 2 * std_dev)) / (4 * std_dev)  # Normalize
        else:
            bb_percent = 0.0
    else:
        bb_percent = 0.0
    features.append(bb_percent)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 15)    # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        features = enhanced_s[123:]  # Retrieve computed features
        if trend_direction > 0:  # Uptrend
            reward += 20 * features[0]  # Positive reward for upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 20 * (1 - features[0])  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        features = enhanced_s[123:]
        reward += 10 * (1 - features[0])  # Reward for mean-reversion potential

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range