import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate daily returns
    closing_prices = s[::6]  # Closing prices (s[i*6 + 0])
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Feature 1: Average daily return over the last 20 days
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0.0
    features.append(avg_daily_return)
    
    # Feature 2: Volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0.0
    features.append(volatility)
    
    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    # Calculate gains and losses
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 15  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 15  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the range [-100, 100]
    return np.clip(reward, -100, 100)