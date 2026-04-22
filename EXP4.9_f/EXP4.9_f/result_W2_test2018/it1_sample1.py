import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    days = len(closing_prices)

    # Feature 1: Average Daily Return over the last 20 days
    daily_returns = np.zeros(days)
    for i in range(1, days):
        daily_returns[i] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1] if closing_prices[i - 1] > 0 else 0
    avg_daily_return = np.mean(daily_returns)

    # Feature 2: Historical Volatility (standard deviation of daily returns)
    historical_volatility = np.std(daily_returns) if days > 1 else 0

    # Feature 3: Price Momentum (current close vs. close 5 days ago)
    price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if days > 5 and closing_prices[5] > 0 else 0

    # Feature 4: Volume Momentum (current volume vs. average volume over the last 20 days)
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    volume_momentum = (volumes[-1] - avg_volume) / avg_volume if avg_volume > 0 else 0

    features = [avg_daily_return, historical_volatility, price_momentum, volume_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    avg_daily_return = features[0]
    historical_volatility = features[1]

    # Calculate dynamic thresholds based on historical volatility
    risk_threshold = np.mean(historical_volatility) + 2 * np.std(historical_volatility)
    trend_threshold = 0.3  # Relative threshold for trend direction

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        if avg_daily_return > 0:  # BUY-aligned feature
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:  # SELL-aligned feature
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > 0 and avg_daily_return > 0:  # Uptrend and upward feature
            reward += 10  # Positive reward
        elif trend_direction < 0 and avg_daily_return < 0:  # Downtrend and downward feature
            reward += 10  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if avg_daily_return < 0:  # Oversold condition
            reward += 10  # Positive reward for mean-reversion buy
        else:  # Overbought condition
            reward += -10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]