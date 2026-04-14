import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    
    # Extract closing prices for simplicity
    closing_prices = s[0::6]  # Closing prices are at indices 0, 6, 12, ..., 114
    
    # Feature 1: 14-day moving average (MA)
    ma_period = 14
    if len(closing_prices) >= ma_period:
        moving_average = np.mean(closing_prices[-ma_period:])
    else:
        moving_average = np.nan  # Handle edge case
    
    # Feature 2: 14-day volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    volatility = np.std(returns[-ma_period:]) if len(returns) >= ma_period else np.nan
    
    # Feature 3: Momentum indicator (Rate of Change)
    if len(closing_prices) >= ma_period:
        momentum = closing_prices[-1] / closing_prices[-ma_period] - 1  # Current price compared to the price 14 days ago
    else:
        momentum = np.nan  # Handle edge case
    
    # Collect features into a list while handling NaNs
    features = [
        moving_average if not np.isnan(moving_average) else 0,
        volatility if not np.isnan(volatility) else 0,
        momentum if not np.isnan(momentum) else 0,
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if enhanced_state[123] > 0:  # Assuming positive features align with BUY
            return np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:
            return np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > 0.4:
        if enhanced_state[123] > 0:  # Assuming positive features align with BUY
            reward += np.random.uniform(-10, -5)  # Moderate negative reward for BUY
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_state[123] > 0:  # Uptrend and positive features
            reward += np.random.uniform(10, 20)  # Positive reward for correct direction
        elif trend_direction < -0.3 and enhanced_state[123] < 0:  # Downtrend and negative features
            reward += np.random.uniform(10, 20)  # Positive reward for correct bearish bet
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_state[123] < 0:  # Negative features for mean reversion
            reward += np.random.uniform(5, 15)  # Reward for mean-reversion features
        else:
            reward -= np.random.uniform(5, 15)  # Penalize breakout-chasing features
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility
    
    return np.clip(reward, -100, 100)  # Ensure reward is within bounds