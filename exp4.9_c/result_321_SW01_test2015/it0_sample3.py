import numpy as np

def revise_state(s):
    # Extract closing prices for calculating features
    closing_prices = s[0::6]  # Closing prices are at indices 0, 6, 12, ..., 114
    N = len(closing_prices)

    # Feature 1: Moving Average (MA)
    window_size = 5
    ma = np.mean(closing_prices[-window_size:]) if N >= window_size else np.nan

    # Feature 2: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-window_size:]) if N >= window_size else np.nan
    avg_loss = np.mean(loss[-window_size:]) if N >= window_size else np.nan
    
    rs = avg_gain / avg_loss if avg_loss > 0 else np.nan  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan

    # Feature 3: Historical Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-window_size:]) if N >= window_size else np.nan

    # Return the computed features as a numpy array
    features = [ma, rsi, volatility]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # Extract regime vector
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        return reward  # Early return as risk is high
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10  # Positive reward for upward features
        else:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward