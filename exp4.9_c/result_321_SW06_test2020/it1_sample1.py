import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    # Feature 1: Average Daily Return over the last 19 days
    avg_daily_return = np.mean(daily_returns[-19:]) if len(daily_returns) >= 19 else 0

    # Feature 2: Volatility (Standard deviation of daily returns)
    volatility = np.std(daily_returns[-19:]) if len(daily_returns) >= 19 else 0

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    gains = (daily_returns[daily_returns > 0]).sum() / 14 if len(daily_returns[daily_returns > 0]) > 0 else 0
    losses = (-daily_returns[daily_returns < 0]).sum() / 14 if len(daily_returns[daily_returns < 0]) > 0 else 0
    rs = gains / losses if losses > 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 4: Trend Strength (absolute value of the last 5-day moving average)
    if len(closing_prices) >= 5:
        trend_strength = np.mean(closing_prices[-5:]) - np.mean(closing_prices[-20:])  # 5-day MA - 20-day MA
    else:
        trend_strength = 0

    # Feature 5: Market Breadth (percentage of stocks above their 50-day moving average)
    # This could be an external input (not from `s`), but assuming we have access to a breadth measure
    breadth = calculate_market_breadth()  # Placeholder function for market breadth, implement separately

    features = [avg_daily_return, volatility, rsi, trend_strength, breadth]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]

    # Calculate historical thresholds for risk and volatility
    historical_volatility = np.std(features[1])  # Assuming features[1] is volatility
    risk_threshold = historical_volatility * 1.5  # Example threshold based on historical data
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40 if features[0] > 0 else 5  # Features[0] is avg_daily_return
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20 if features[0] > 0 else 0

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive return
            reward += 20  # Positive reward for following the trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative return
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Oversold condition
            reward += 15  # Reward for buying
        elif features[2] > 70:  # Overbought condition
            reward += -15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clipping the reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward

def calculate_market_breadth():
    # Placeholder function to calculate market breadth
    # This could involve checking how many stocks are above their moving averages
    # Currently returns a dummy value; implement appropriate logic based on available data
    return 0.5  # Example static value, replace with real breadth calculation