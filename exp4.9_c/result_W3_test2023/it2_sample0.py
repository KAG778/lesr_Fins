import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    features = []

    # Feature 1: Rate of Change (ROC) over the last 10 days
    roc = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if len(closing_prices) > 10 else 0
    features.append(roc)

    # Feature 2: 14-day Average True Range (ATR) for volatility measurement
    high_prices = s[2::6]
    low_prices = s[3::6]
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:]) - np.minimum(low_prices[1:], closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    # Feature 3: Z-Score of the last closing price based on historical data
    historical_mean = np.mean(closing_prices)
    historical_std = np.std(closing_prices)
    z_score = (closing_prices[-1] - historical_mean) / historical_std if historical_std != 0 else 0
    features.append(z_score)

    # Feature 4: 10-day Historical Volatility based on returns
    if len(closing_prices) >= 10:
        returns = np.diff(closing_prices) / closing_prices[:-1]
        historical_volatility = np.std(returns[-10:])  # Rolling standard deviation over the last 10 days
    else:
        historical_volatility = 0.0
    features.append(historical_volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s[123:]) > 0 else 1  # Avoid division by zero
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Oversold condition
            reward += 15  # Reward for buying
        elif enhanced_s[123] > 0:  # Overbought condition
            reward -= 15  # Penalize for selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]