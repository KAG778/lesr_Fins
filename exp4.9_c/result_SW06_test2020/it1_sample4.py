import numpy as np

def revise_state(s):
    features = []

    closing_prices = s[0::6]  # Extract closing prices (every 6th element)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Feature 1: Average Daily Return over the last 20 days
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0.0
    features.append(avg_daily_return)

    # Feature 2: Volatility using the last 20 days of returns
    volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0.0
    features.append(volatility)

    # Feature 3: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0.0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0.0
    macd = short_ema - long_ema
    features.append(macd)

    # Feature 4: Relative Strength Index (RSI)
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for risk
    historical_std = np.std(enhanced_s[123:])  # Assume features start at index 123
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward += -np.random.uniform(30, 50)  # Strong negative for BUY
    elif risk_level > risk_threshold_medium:
        reward += -np.random.uniform(5, 15)  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]