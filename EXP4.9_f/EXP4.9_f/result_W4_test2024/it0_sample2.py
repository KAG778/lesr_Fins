import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Initialize an empty list for features
    features = []
    
    # Compute the daily returns based on closing prices
    closing_prices = s[::6]  # Extract closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns calculation
    features.append(np.mean(daily_returns))  # Mean daily return
    
    # Calculate the volatility (standard deviation of daily returns)
    if len(daily_returns) > 0:
        volatility = np.std(daily_returns)  
    else:
        volatility = 0
    features.append(volatility)  # Volatility of returns
    
    # Calculate the relative strength index (RSI) over the last 14 days for momentum
    if len(daily_returns) >= 14:
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.nan  # Not enough data for RSI
    features.append(rsi)  # RSI feature
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features

    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Strong negative for risky buys
        # Mild positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10)   # Mild positive for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Moderate penalty for risky buys

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += np.random.uniform(10, 20)  # Reward for correct bullish bet
        elif trend_direction < -0.3:
            reward += np.random.uniform(10, 20)  # Reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (not implemented here)
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds