import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)

    # Reshape the raw state into a 20-day view with 6 features each
    days = s.reshape((20, 6))

    # Feature 1: Calculate the daily returns
    closing_prices = days[:, 0]  # Closing prices
    daily_returns = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices  # daily return as a percentage
    avg_daily_return = np.mean(daily_returns)  # Average daily return

    # Feature 2: Calculate the volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns)

    # Feature 3: Calculate the Relative Strength Index (RSI) over the last 14 days
    # Use the last 14 days for RSI calculation, ensuring we handle edge cases
    if len(daily_returns) < 14:
        rsi = np.nan  # Not enough data
    else:
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss > 0 else np.nan
        rsi = 100 - (100 / (1 + rs))  # RSI formula

    # Feature 4: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:])  # Short-term EMA (12-day)
    long_ema = np.mean(closing_prices[-26:])  # Long-term EMA (26-day)
    macd = short_ema - long_ema  # MACD value

    # Combine features into a numpy array
    features = [avg_daily_return, volatility, rsi, macd]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    
    reward = 0.0  # Initialize reward

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming feature[0] indicates a BUY signal
            return np.random.uniform(-50, -30)  # Strong negative reward
        # Mild positive reward for SELL-aligned features
        return np.random.uniform(5, 10)  # Mild positive reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Assuming feature[0] indicates a BUY signal
            return np.random.uniform(-20, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and BUY signal
            reward += 10  # Positive reward for correct direction
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and SELL signal
            reward += 10  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming feature[0] indicates a SELL signal
            reward += 5  # Reward mean-reversion features
        else:
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward