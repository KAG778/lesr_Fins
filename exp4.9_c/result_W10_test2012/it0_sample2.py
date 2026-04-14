import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Compute daily returns
    closing_prices = s[::6]  # Closing prices (s[i*6 + 0])
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.concatenate(([0], daily_returns))  # Pad with zero for the first day
    features.append(np.mean(daily_returns))  # Mean daily return
    
    # Compute volatility as standard deviation of daily returns
    volatility = np.std(daily_returns)
    features.append(volatility)
    
    # Calculate the relative strength index (RSI) for the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
    rs = avg_gain / avg_loss if avg_loss > 0 else 0  # Avoid division by zero
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
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for risky BUY-aligned features
        reward = -np.random.uniform(30, 50)  # Strong negative reward for BUY
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # If risk is low, check for trends
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:
                reward += np.random.uniform(10, 20)  # Positive reward for upward features
            else:
                reward += np.random.uniform(10, 20)  # Positive reward for downward features

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Assuming we have some features in enhanced_s[123:]
            # Use the RSI as a mean-reversion indicator for example
            rsi = enhanced_s[123][2]  # Assuming RSI is the third feature
            if rsi < 30:
                reward += np.random.uniform(10, 20)  # Buy signal when oversold
            elif rsi > 70:
                reward += np.random.uniform(10, 20)  # Sell signal when overbought

    # Priority 4 — HIGH VOLATILITY (no crisis)
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward