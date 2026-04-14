import numpy as np

def revise_state(s):
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Volume Change (percentage change from previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if len(volumes) > 1 and volumes[-2] != 0 else 0

    # Feature 3: 14-day Relative Strength Index (RSI) calculation
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0

    rs = average_gain / (average_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    # Feature 4: Bollinger Bands Width (to gauge volatility)
    rolling_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bollinger_width = (rolling_std / rolling_mean) if rolling_mean != 0 else 0

    # Combine features into a single array and return
    features = [price_momentum, volume_change, rsi, bollinger_width]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Adaptive thresholds based on historical volatility
    historical_std = np.std(enhanced_s[0:120])  # Use raw state for historical volatility assessment.
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # STRONG NEGATIVE reward for risky BUY-aligned features
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        if trend_direction > 0:
            reward += 10 if enhanced_s[123] > 0 else 0  # Reward BUY aligned with upward trend
        else:
            reward += 10 if enhanced_s[123] < 0 else 0  # Reward SELL aligned with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < medium_risk_threshold:
        reward += 10 if enhanced_s[123] < 0 else -10  # Reward mean-reversion or penalize breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]