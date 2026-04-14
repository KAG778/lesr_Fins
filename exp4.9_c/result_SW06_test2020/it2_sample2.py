import numpy as np

def revise_state(s):
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices from raw state
    high_prices = s[2::6]     # High prices from raw state
    low_prices = s[3::6]      # Low prices from raw state

    # Feature 1: Average True Range (ATR) for volatility measurement (14-day)
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0

    # Feature 2: Momentum (Rate of Change) over the last 10 days
    momentum = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 11 else 0.0

    # Feature 3: Percentage drop from recent peak (Crisis Indicator)
    recent_peak = np.max(closing_prices[-20:])  # Look at the last 20 days for peak
    crisis_indicator = (recent_peak - closing_prices[-1]) / recent_peak if recent_peak != 0 else 0.0

    # Feature 4: Z-score of returns to assess extreme movements
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    mean_return = np.mean(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    std_return = np.std(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0

    features = [atr, momentum, crisis_indicator, z_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(enhanced_s[123:])  # Using features for risk assessment
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(5, 15)  # Mild negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Positive reward for upward trend
        else:
            reward += np.random.uniform(10, 20)  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < risk_threshold_medium:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion strategies

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]