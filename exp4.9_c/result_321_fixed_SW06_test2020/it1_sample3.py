import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    # Feature 1: 20-day Moving Average of Returns
    moving_avg_returns = np.mean(returns[-20:]) if len(returns) >= 20 else 0.0

    # Feature 2: 20-day Volatility (Standard Deviation of Returns)
    moving_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.0

    # Feature 3: Price Range (High - Low) over the last 20 days
    price_range = np.max(closing_prices) - np.min(closing_prices)

    # Feature 4: Volume Spike (Current Volume vs. 20-day Average)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0.0
    volume_spike = volumes[-1] / avg_volume if avg_volume != 0 else 0.0

    return np.array([moving_avg_returns, moving_volatility, price_range, volume_spike])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Determine relative thresholds based on historical volatility
    historical_std = np.std(features[1:])  # Using the moving volatility feature for dynamic thresholds
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std
    high_volatility_threshold = 0.6 * historical_std

    # Priority 1: Risk Management
    if risk_level > high_risk_threshold:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 5.0 if features[0] < 0 else 0  # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if features[0] > 0:  # Positive momentum
            reward += 10.0 * features[0]  # Reward for momentum alignment
        else:  # Negative momentum
            reward += -10.0 * features[0]  # Penalize if betting against trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 5.0  # Reward for potential buying opportunity
        elif features[0] > 0.01:  # Overbought condition
            reward += -5.0  # Reward for potential selling opportunity

    # Priority 4: High Volatility
    if volatility_level > high_volatility_threshold and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))