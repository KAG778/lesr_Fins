import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    features = []

    # Feature 1: 10-day Historical Volatility
    if len(closing_prices) >= 10:
        returns = np.diff(closing_prices) / closing_prices[:-1]
        historical_volatility = np.std(returns[-10:])  # Rolling standard deviation over the last 10 days
    else:
        historical_volatility = 0.0
    features.append(historical_volatility)

    # Feature 2: Price Change Percentage (last day)
    if len(closing_prices) >= 2:
        price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    else:
        price_change = 0.0
    features.append(price_change)

    # Feature 3: Volume Spike Detection
    if len(volumes) >= 2:
        avg_volume = np.mean(volumes[:-1])
        volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    else:
        volume_change = 0.0
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive for upward alignment
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Assuming oversold situation (mean reversion BUY)
            reward += 15  # Reward mean-reversion BUY
        elif enhanced_s[123] > 0:  # Overbought situation (mean reversion SELL)
            reward += 15  # Reward mean-reversion SELL

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]