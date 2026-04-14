import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices (every 6th element)
    
    # Feature 1: Average True Range (ATR) for volatility measurement (14-day)
    highs = s[2::6]  # High prices
    lows = s[3::6]   # Low prices
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:]))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0
    features.append(atr)

    # Feature 2: Momentum (Rate of Change) over the last 10 days
    momentum = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 10 and closing_prices[-11] != 0 else 0.0
    features.append(momentum)

    # Feature 3: Z-score of recent returns (to capture mean reversion)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    mean_return = np.mean(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    std_return = np.std(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score)

    # Feature 4: Percentage of Last Close to 20-day High
    last_close = closing_prices[-1]
    high_20 = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    percent_close_to_high = last_close / high_20 if high_20 != 0 else 0.0
    features.append(percent_close_to_high)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical volatility
    historical_std = np.std(enhanced_s[123:])  # Utilizing features for risk assessment
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_low = 0.4 * historical_std
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(10, 20)   # Mild positive for SELL
    elif risk_level > risk_threshold_low:
        reward -= np.random.uniform(5, 15)    # Moderate negative for BUY

    # Priority 2: Trend Following (momentum alignment)
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_low:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward trend
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < risk_threshold_low:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)   # Penalize breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_low:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the range [-100, 100]
    return np.clip(reward, -100, 100)