import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate daily returns
    closing_prices = s[::6]  # Extract closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Handle edge case: if there's no previous price, return 0 for daily return
    daily_returns = np.append(np.nan, daily_returns)  # Add NaN for the first day
    daily_returns = np.nan_to_num(daily_returns)  # Replace NaN with 0

    # Feature 1: Average Daily Return
    avg_daily_return = np.mean(daily_returns)

    # Feature 2: Volatility (Standard Deviation of Returns)
    volatility = np.std(daily_returns)

    # Feature 3: Moving Average of Closing Prices (20-day)
    moving_average = np.mean(closing_prices)

    # Feature 4: Relative Strength Index (RSI) - 14-day
    gain = np.where(daily_returns > 0, daily_returns, 0)
    loss = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain[-14:]) > 0 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss[-14:]) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    features.extend([avg_daily_return, volatility, moving_average, rsi])
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
        # Strong negative reward for BUY-aligned features
        reward += -40  # Example strong negative reward
        # Mild positive reward for SELL-aligned features
        reward += 7   # Example mild positive reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example moderate negative reward

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Reward for upward features
            reward += 15  # Example positive reward
        elif trend_direction < -0.3:
            # Positive reward for downward features (correct bearish bet)
            reward += 15  # Example positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 10  # Example positive reward for mean reversion
        # Penalize breakout-chasing features
        reward += -5   # Example mild penalty

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        # Reduce reward magnitude by 50%
        reward *= 0.5

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds