import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []

    # Extract closing prices
    closing_prices = s[0::6]

    # Calculate daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0])
    mean_return = np.mean(daily_returns)  # Mean return
    volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0  # Volatility (standard deviation of returns)

    # Calculate momentum over a 5-day window
    n_days = 5
    momentum = (closing_prices[-1] - closing_prices[-n_days]) / closing_prices[-n_days] if len(closing_prices) > n_days else 0

    # Calculate RSI
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 0  # Not enough data for RSI

    # Calculate Average True Range (ATR) for volatility measurement
    highs = s[2::6]
    lows = s[3::6]
    true_ranges = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR

    # Create features list
    features = [mean_return, volatility, momentum, rsi, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract the features
    reward = 0.0

    # Dynamic thresholds for risk management
    risk_threshold = 0.7
    if risk_level > risk_threshold:
        # Strong negative reward for BUY-aligned features
        if features[2] > 0:  # Assuming momentum is positive
            reward -= np.random.uniform(40, 60)
        else:  # Mild positive reward for SELL-aligned features
            reward += np.random.uniform(10, 20)
    
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[2] > 0:
            reward -= np.random.uniform(20, 30)

    # Priority 2 — TREND FOLLOWING (only when risk is low)
    if risk_level < 0.4:
        if abs(trend_direction) > 0.3:
            if trend_direction > 0 and features[2] > 0:  # Uptrend and positive momentum
                reward += np.random.uniform(15, 30)
            elif trend_direction < 0 and features[2] < 0:  # Downtrend and negative momentum
                reward += np.random.uniform(15, 30)

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            if features[3] < 30:  # RSI < 30 indicates oversold
                reward += np.random.uniform(15, 25)
            elif features[3] > 70:  # RSI > 70 indicates overbought
                reward += np.random.uniform(15, 25)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the bounds
    return max(-100, min(100, reward))