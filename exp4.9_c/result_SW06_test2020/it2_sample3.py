import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices from the raw state
    closing_prices = s[0::6]  # Closing prices
    
    # Feature 1: Exponential Moving Average (EMA) difference (short vs long)
    ema_short = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0.0  # 10-day EMA
    ema_long = np.mean(closing_prices[-50:]) if len(closing_prices) >= 50 else 0.0  # 50-day EMA
    features.append(ema_short - ema_long)  # Difference to indicate trend strength
    
    # Feature 2: Average True Range (ATR) for volatility measurement (14-day)
    highs = s[2::6]
    lows = s[3::6]
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1])
    tr = np.maximum(tr, closing_prices[:-1] - lows[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0
    features.append(atr)
    
    # Feature 3: Z-score of recent returns for mean-reversion signal
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    mean_return = np.mean(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    std_return = np.std(daily_returns[-14:]) if len(daily_returns) >= 14 else 0
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score)

    # Feature 4: Momentum (rate of change over the last 5 days)
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 6 else 0.0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds based on features
    historical_std = np.std(enhanced_s[123:])  # Assuming features start at index 123
    risk_threshold_high = 1.5 * historical_std
    risk_threshold_low = 0.5 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(10, 20)   # Mild positive for SELL
    elif risk_level > risk_threshold_low:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_low:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for upward trend
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < risk_threshold_low:
        reward += 5  # Reward mean-reversion features
        reward -= 5  # Penalize breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_low:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the range [-100, 100]
    return np.clip(reward, -100, 100)