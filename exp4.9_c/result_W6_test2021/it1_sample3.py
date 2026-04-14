import numpy as np

def revise_state(s):
    features = []

    # Feature 1: Price Momentum (percentage change over the last 5 days)
    try:
        price_momentum = (s[114] - s[109]) / s[109]  # (Close day 19 - Close day 14) / Close day 14
    except ZeroDivisionError:
        price_momentum = 0.0
    features.append(price_momentum)

    # Feature 2: Average Volume over the last 10 days
    avg_volume = np.mean(s[4:120:6][-10:])  # Average volume for the last 10 days
    features.append(avg_volume)

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0).mean()
        loss = np.abs(np.where(delta < 0, delta, 0)).mean()
        rs = gain / loss if loss > 0 else 0
        return 100 - (100 / (1 + rs))

    closing_prices = s[0::6][-14:]  # Last 14 closing prices
    rsi = calculate_rsi(closing_prices)
    features.append(rsi)

    # Feature 4: Historical Volatility (Standard deviation of closing prices over the last 20 days)
    historical_volatility = np.std(s[0:120:6][-20:])  # Standard deviation of closing prices
    features.append(historical_volatility)

    # Feature 5: Price vs. Moving Average (20-day moving average)
    moving_average = np.mean(s[0:120:6][-20:])  # 20-day moving average
    price_vs_ma = (s[114] - moving_average) / moving_average if moving_average != 0 else 0
    features.append(price_vs_ma)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate threshold values based on historical data
    mean_risk_level = 0.4  # Example: compute from historical data
    std_risk_level = 0.2   # Example: compute from historical data

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > mean_risk_level + 1.5 * std_risk_level:  # High risk
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > mean_risk_level:
        reward -= 20  # Moderate penalty for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= mean_risk_level:  # Low risk
        if trend_direction > 0:
            reward += 20  # Reward for upward momentum
        else:
            reward += 20  # Reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < mean_risk_level:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > (mean_risk_level + 1.5 * std_risk_level) and risk_level < mean_risk_level:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward