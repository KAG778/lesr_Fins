import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Feature 1: Average daily return over the last 20 days
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0.0
    
    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0.0
    
    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 4: Momentum (difference between the last closing price and the 10-day moving average)
    moving_average_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0.0
    momentum = closing_prices[-1] - moving_average_10
    
    # Feature 5: Price Range (High - Low for each day)
    high_prices = s[2::6]  # High prices from raw state
    low_prices = s[3::6]   # Low prices from raw state
    price_range = np.mean(high_prices[-20:]) - np.mean(low_prices[-20:]) if len(high_prices) >= 20 else 0.0
    
    features = [avg_daily_return, volatility, rsi, momentum, price_range]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Relative thresholds based on historical std deviations
    mean_risk_threshold = 0.5  # Placeholder for historical mean risk level
    std_risk_threshold = 0.2    # Placeholder for historical std of risk level
    
    # Priority 1: Risk Management
    if risk_level > mean_risk_threshold + 1.5 * std_risk_threshold:
        reward += -np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY
    elif risk_level > mean_risk_threshold + 0.5 * std_risk_threshold:
        reward += -np.random.uniform(10, 20)  # MODERATE NEGATIVE for BUY
    
    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < mean_risk_threshold:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for upward trend
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < mean_risk_threshold - 0.5 * std_risk_threshold:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > mean_risk_threshold + 1.2 * std_risk_threshold and risk_level < mean_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the range [-100, 100]
    return float(np.clip(reward, -100, 100))