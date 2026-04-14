import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []
    
    # Calculate daily returns
    closing_prices = s[0::6]  # Closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    daily_returns = np.concatenate(([0], daily_returns))  # Prepend 0 for alignment
    features.append(np.mean(daily_returns))  # Average daily return

    # Calculate volatility as standard deviation of returns
    volatility = np.std(daily_returns)
    features.append(volatility if volatility != 0 else 0)  # Avoid division by zero

    # Calculate relative strength index (RSI)
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    
    avg_gain = np.mean(gains[-14:])  # Look-back window of 14 days
    avg_loss = np.mean(losses[-14:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI calculation
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
        if enhanced_s[123] == 0:  # Assuming BUY-aligned features are at index 123
            reward = np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY
        else:
            reward = np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        if enhanced_s[123] == 0:  # Assuming BUY-aligned features are at index 123
            reward = np.random.uniform(-20, -10)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            if enhanced_s[123] == 1:  # Assuming upward features are at index 123
                reward += 10  # Positive reward for correct trend-following
        else:  # Downtrend
            if enhanced_s[123] == 2:  # Assuming downward features are at index 123
                reward += 10  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] == 3:  # Assuming mean-reversion features are at index 123
            reward += 10  # Reward for mean-reversion features
        else:
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY (no crisis)
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds