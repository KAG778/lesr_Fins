import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices (every 6th element)
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    
    # Feature 1: Average True Range (ATR) for measuring volatility (14-day)
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                               abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0
    features.append(atr)
    
    # Feature 2: Rate of Change (Momentum) over the last 10 days
    momentum = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if closing_prices[-11] != 0 else 0.0
    features.append(momentum)

    # Feature 3: Crisis Indicator (percentage drop from recent peak over the last 20 days)
    recent_peak = np.max(closing_prices[-20:])  # Look at the last 20 days for peak
    crisis_indicator = (recent_peak - closing_prices[-1]) / recent_peak if recent_peak != 0 else 0.0
    features.append(crisis_indicator)

    # Feature 4: Z-score of returns for mean-reversion signal
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    mean_return = np.mean(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    std_return = np.std(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds based on the standard deviation of existing features
    historical_std = np.std(enhanced_s[123:])  # Assuming features start at index 123
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward += -np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(10, 20)   # Mild positive for SELL
    elif risk_level > risk_threshold_moderate:
        reward += -np.random.uniform(5, 15)   # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Positive reward for upward trend
        else:
            reward += np.random.uniform(10, 20)  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        reward += np.random.uniform(5, 15)   # Reward mean-reversion
        reward -= np.random.uniform(5, 15)    # Penalize for breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]