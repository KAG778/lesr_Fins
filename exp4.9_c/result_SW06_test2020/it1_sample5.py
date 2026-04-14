import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (every 6th element)
    
    # Feature 1: Average daily return over the last 20 days
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0.0
    
    # Feature 2: Volatility (standard deviation of daily returns over the last 20 days)
    volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0.0
    
    # Feature 3: 14-day Relative Strength Index (RSI)
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 4: Momentum (rate of change over the last 10 days)
    momentum = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 10 else 0.0
    
    # Feature 5: 5-day moving average convergence/divergence (MACD)
    ema_short = np.mean(closing_prices[-12:-2])  # Short EMA (12 days)
    ema_long = np.mean(closing_prices[-26:-6])   # Long EMA (26 days)
    macd = ema_short - ema_long
    
    # Combine features into an array
    features = [avg_daily_return, volatility, rsi, momentum, macd]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds for dynamic risk levels
    historical_std = np.std(enhanced_s[123:])  # Utilizing features for risk assessment
    risk_threshold_1 = historical_std * 1.5  # High risk
    risk_threshold_2 = historical_std * 1.2  # Moderate risk
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1: Risk Management
    if risk_level > risk_threshold_1:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
    elif risk_level > risk_threshold_2:
        reward -= np.random.uniform(10, 20)  # Mild negative for BUY
    
    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_2:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for upward trends
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Positive reward for downward trends
    
    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < historical_std:
        reward += 5  # Reward mean-reversion features
        reward -= 3  # Mild penalty for breakout chasing
    
    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_2:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    # Ensure reward is within the range [-100, 100]
    return np.clip(reward, -100, 100)