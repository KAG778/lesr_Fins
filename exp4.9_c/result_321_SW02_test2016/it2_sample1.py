import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]          # Trading volumes

    # Feature 1: 10-day Momentum (current close vs. close 10 days ago)
    momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0
    features.append(momentum)

    # Feature 2: Exponential Moving Average (EMA) of closing prices
    if len(closing_prices) >= 10:
        weights = np.exp(np.linspace(-1., 0., 10))
        weights /= weights.sum()
        ema = np.dot(weights, closing_prices[-10:])
    else:
        ema = closing_prices[-1]
    features.append(ema)

    # Feature 3: Z-score of Returns (last 20 days)
    if len(closing_prices) >= 20:
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        z_score_returns = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    else:
        z_score_returns = 0
    features.append(z_score_returns)

    # Feature 4: Average True Range (ATR) for volatility measure
    high_prices = s[1:120:6]  # High prices for 20 days
    low_prices = s[2:120:6]   # Low prices for 20 days
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    mean_risk = 0.5  # Placeholder for historical mean risk level
    std_risk = 0.2   # Placeholder for historical std for risk level
    risk_threshold = mean_risk + 1 * std_risk  # Example threshold based on std deviation

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for risky BUY signals
        reward += np.random.uniform(5, 10)  # Mild positive for SELL signals
        return np.clip(reward, -100, 100)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < mean_risk:
        if trend_direction > 0:
            reward += np.random.uniform(10, 25)  # Positive reward for bullish momentum
        else:
            reward += np.random.uniform(10, 25)  # Positive reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Positive reward for mean-reversion features
        reward -= np.random.uniform(5, 15)  # Negative for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < mean_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]