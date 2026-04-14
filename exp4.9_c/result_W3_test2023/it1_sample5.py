import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Average True Range (ATR) over the last 14 days
    if len(closing_prices) >= 14:
        high_low = np.array(s[2::6]) - np.array(s[3::6])  # High - Low
        high_close = np.abs(np.array(s[2::6]) - np.roll(closing_prices, 1)[1:])  # High - Previous Close
        low_close = np.abs(np.array(s[3::6]) - np.roll(closing_prices, 1)[1:])  # Low - Previous Close
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = np.mean(true_range[-14:])  # ATR
    else:
        atr = 0.0

    # Feature 2: Skewness of returns
    returns = np.diff(closing_prices) / closing_prices[:-1]
    skewness = np.mean((returns - np.mean(returns))**3) / (np.std(returns, ddof=1)**3) if len(returns) > 2 else 0  # Sample skewness

    # Feature 3: Z-score of the last closing price compared to the mean over the last 20 days
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:], ddof=1)
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    else:
        z_score = 0.0

    features = [atr, skewness, z_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds for reward calculations
    historical_std = np.std(enhanced_s[0:120])  # Historical standard deviation based on the raw state
    high_risk_threshold = 0.7 * historical_std
    mid_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
    elif risk_level > mid_risk_threshold:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < mid_risk_threshold:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < mid_risk_threshold:
        if enhanced_s[123] < 0:  # Assuming oversold feature
            reward += 10  # Reward mean-reversion features
        else:  # Assuming overbought feature
            reward -= 10  # Penalize for breakout-chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < mid_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]