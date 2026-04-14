import numpy as np

def revise_state(s):
    # Features array to hold the new features
    features = []
    
    # Extracting closing prices from the raw state
    closing_prices = s[0::6]  # Closing prices
    
    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-10:])  # 10-day EMA
    ema_long = np.mean(closing_prices[-50:])   # 50-day EMA
    features.append(ema_short - ema_long)  # Difference to indicate trend strength
    
    # Feature 2: Volatility (standard deviation of closing prices over the last 20 days)
    volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(volatility)

    # Feature 3: Average True Range (ATR) for measuring market volatility
    highs = s[2::6]
    lows = s[3::6]
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0
    features.append(atr)

    # Feature 4: Percentage of Last Close to 20-day High
    last_close = closing_prices[-1]
    high_20 = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    if high_20 != 0:
        percent_close_to_high = last_close / high_20
    else:
        percent_close_to_high = 0.0
    features.append(percent_close_to_high)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical volatility
    risk_thresholds = {
        "high": 0.7,
        "medium": 0.4,
        "low": 0.1
    }

    # Initialize reward
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_thresholds["high"]:
        reward = -np.random.uniform(30, 50)  # Strong negative for BUY
    elif risk_level > risk_thresholds["medium"]:
        reward = -np.random.uniform(5, 15)  # Mild negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < risk_thresholds["medium"]:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward trend
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < risk_thresholds["low"]:
        reward += 5  # Reward mean-reversion features
        reward -= 5  # Penalize breakout-chasing features

    # Priority 4: Adjust for High Volatility
    if volatility_level > 0.6 and risk_level < risk_thresholds["medium"]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the range [-100, 100]
    return np.clip(reward, -100, 100)