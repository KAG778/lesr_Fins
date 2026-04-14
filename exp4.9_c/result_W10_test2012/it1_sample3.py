import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []

    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices (s[i*6 + 0])
    
    # Feature 1: Mean of the last 20 daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.concatenate(([0], daily_returns))  # Pad with zero for the first day
    mean_daily_return = np.mean(daily_returns[-20:])  # Last 20 days
    features.append(mean_daily_return)

    # Feature 2: Volatility (standard deviation of daily returns) over the last 20 days
    volatility = np.std(daily_returns[-20:])  # Last 20 days
    features.append(volatility)

    # Feature 3: Relative strength index (RSI) for the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rs = avg_gain / avg_loss if avg_loss > 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 4: Momentum indicator (difference between latest closing price and a 20-day moving average)
    moving_avg_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    momentum = closing_prices[-1] - moving_avg_20
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_volatility = np.std(enhanced_s[123:][1])  # Assuming the second feature is volatility
    historical_rsi = np.mean(enhanced_s[123:][2])  # Assuming the third feature is RSI

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward = -50 if enhanced_s[123][3] > 0 else 10  # Assuming the fourth feature is momentum
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20 if enhanced_s[123][3] > 0 else 5  # Assuming the fourth feature is momentum

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Reward momentum alignment
        if trend_direction > 0 and enhanced_s[123][3] > 0:  # Assuming the fourth feature is momentum
            reward += 30  # Positive reward for aligning with upward momentum
        elif trend_direction < 0 and enhanced_s[123][3] <= 0:  # Assuming the fourth feature is momentum
            reward += 30  # Positive reward for aligning with downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward for mean-reverting behavior
        if enhanced_s[123][2] < 30:  # Assuming RSI is the third feature
            reward += 20  # Buy signal when oversold
        elif enhanced_s[123][2] > 70:  # Assuming RSI is the third feature
            reward -= 20  # Penalize for buying in an overbought market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward