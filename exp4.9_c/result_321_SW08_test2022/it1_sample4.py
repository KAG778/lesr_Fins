import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))
    closing_prices = days[:, 0]  # Extract closing prices
    volumes = days[:, 4]          # Extract trading volumes

    # Feature 1: Average Daily Return
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices
    avg_daily_return = np.mean(daily_returns)
    features.append(avg_daily_return)

    # Feature 2: Volatility (Standard Deviation of Daily Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Relative Strength Index (RSI) calculation (14-day)
    if len(daily_returns) >= 14:
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.nan  # Not enough data
    features.append(rsi)

    # Feature 4: Price Momentum (Current Close - 5 Days Ago)
    momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(momentum)

    # Feature 5: Volume Change (Current Volume vs. Previous Volume)
    volume_change = (volumes[-1] / volumes[-2] - 1) if len(volumes) > 1 and volumes[-2] != 0 else 0
    features.append(volume_change)

    # Feature 6: MACD (Moving Average Convergence Divergence)
    short_ema = np.mean(closing_prices[-12:])  # Short-term EMA (12-day)
    long_ema = np.mean(closing_prices[-26:])   # Long-term EMA (26-day)
    macd = short_ema - long_ema
    features.append(macd)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Grab the features from the enhanced state
    reward = 0.0  # Initialize reward

    # Calculate relative thresholds based on the historical data (assuming we have access to historical std)
    historical_std = np.std(features)  # This could be replaced by a more sophisticated calculation

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[3] > 0:  # Assuming feature[3] indicates a BUY signal (momentum)
            return np.random.uniform(-50, -30)  # Strong negative reward
        # Mild positive reward for SELL-aligned features
        return np.random.uniform(5, 10)  # Mild positive reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[3] > 0:  # Uptrend and positive momentum
            reward += 10  # Positive reward
        elif trend_direction < -0.3 and features[3] < 0:  # Downtrend and negative momentum
            reward += 10  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 0:  # Assuming feature[3] indicates a SELL signal
            reward += 5  # Reward mean-reversion features
        else:
            reward -= 5  # Penalize chasing trends

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward