import numpy as np

def revise_state(s):
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

    # Feature 4: Price Momentum (Current Close - Previous Close)
    momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0
    features.append(momentum)

    # Feature 5: Drawdown from the highest price in the last 20 days
    max_price = np.max(closing_prices)
    drawdown = (max_price - closing_prices[-1]) / max_price if max_price > 0 else 0
    features.append(drawdown)

    # Feature 6: Exponential Moving Average Convergence Divergence (MACD)
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

    # Calculate relative thresholds based on historical features
    historical_std = np.std(features)  # Standard deviation of features

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[3] > 0:  # Assuming feature[3] indicates a BUY signal (momentum)
            reward += -50  # Strong penalty for risky BUY
        # Mild positive reward for SELL-aligned features
        reward += 10  # Encouragement to sell
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Moderate penalty for elevated risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[3] > 0:  # Uptrend and positive momentum
            reward += 20  # Strong positive reward for correct direction
        elif trend_direction < -0.3 and features[3] < 0:  # Downtrend and negative momentum
            reward += 20  # Strong positive reward for correct direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3:
        if risk_level < 0.3:
            reward += 15  # Reward for mean-reversion actions when risk is low
        else:
            reward -= 10  # Penalize for chasing trends

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward