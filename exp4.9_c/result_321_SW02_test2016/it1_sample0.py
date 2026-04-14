import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]          # Extracting volumes

    # Feature 1: 5-day Rate of Change (momentum indicator)
    # This captures short-term momentum
    roc_5day = (closing_prices[-1] / closing_prices[-6]) - 1 if len(closing_prices) >= 6 else 0
    features.append(roc_5day)

    # Feature 2: 20-day Average Volume (to capture liquidity)
    avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1  # Avoid division by zero
    current_volume = volumes[-1] if len(volumes) > 0 else 1
    volume_ratio = (current_volume - avg_volume_20) / avg_volume_20
    features.append(volume_ratio)

    # Feature 3: 14-day Relative Strength Index (RSI)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Get last 14 days
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50.0  # Default to neutral RSI if insufficient data
    features.append(rsi)

    # Feature 4: Volatility (standard deviation of returns over 20 days)
    if len(closing_prices) >= 20:
        returns = np.diff(closing_prices[-20:]) / closing_prices[-20:-1]  # Daily returns
        volatility = np.std(returns)
    else:
        volatility = 0
    features.append(volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risks and trends based on historical std
    std_risk_threshold = np.std(enhanced_s[123:]) * 0.5  # Historical risk level threshold
    std_trend_threshold = np.std(enhanced_s[123:]) * 0.3  # Historical trend level threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > std_risk_threshold:
        reward -= 50  # Strong negative reward for buying in high-risk conditions
        reward += 10  # Mild positive reward for selling in high-risk conditions
        return np.clip(reward, -100, 100)  # Exit early if high risk

    elif risk_level > std_risk_threshold * 0.7:
        reward -= 20  # Moderate negative for buying in elevated risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > std_trend_threshold and risk_level < std_risk_threshold * 0.7:
        if trend_direction > 0:
            reward += 20  # Reward for bullish trends
        else:
            reward += 20  # Reward for bearish trends

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < std_trend_threshold and risk_level < std_risk_threshold * 0.5:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < std_risk_threshold * 0.7:
        reward *= 0.5  # Reduce reward magnitude by 50% during high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]