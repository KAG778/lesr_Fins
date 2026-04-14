import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices
    closing_prices = s[0::6]
    
    # Feature 1: Daily Returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_returns = np.concatenate(([0], daily_returns))  # Align lengths
    features.append(np.mean(daily_returns))  # Average daily return

    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns)
    features.append(volatility if volatility != 0 else 0)

    # Feature 3: Skewness of Returns
    skewness = np.mean((daily_returns - np.mean(daily_returns))**3) / (volatility**3) if volatility != 0 else 0
    features.append(skewness)

    # Feature 4: Kurtosis of Returns
    kurtosis = np.mean((daily_returns - np.mean(daily_returns))**4) / (volatility**4) - 3 if volatility != 0 else 0  # Excess kurtosis
    features.append(kurtosis)

    # Feature 5: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(enhanced_s[123:])  # Assuming features start at index 123
    low_risk_threshold = 0.4 * historical_volatility
    high_risk_threshold = 0.7 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY signals
        reward -= 40  # Strong penalty for buying
    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        reward -= 20  # Moderate penalty for buying

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Uptrend
            reward += 15  # Positive reward for upward momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 15  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds