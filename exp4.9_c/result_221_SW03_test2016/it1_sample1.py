import numpy as np

def revise_state(s):
    # Extracting the closing prices for the past 20 days
    closing_prices = s[0:120:6]
    volumes = s[4:120:6]
    
    # Feature 1: 5-Day Moving Average of Prices
    ma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    
    # Feature 2: Exponential Moving Average (EMA) - 12 Days
    ema_12 = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else np.nan
    
    # Feature 3: Price Momentum (closing price difference with 5-Day MA)
    momentum = closing_prices[-1] - ma_5 if not np.isnan(ma_5) else 0
    
    # Feature 4: Relative Strength Index (RSI) - 14 days
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0).mean() if len(deltas) > 0 else 0
    loss = -np.where(deltas < 0, deltas, 0).mean() if len(deltas) > 0 else 0
    rs = gain / loss if loss != 0 else np.nan
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 5: Volatility as Standard Deviation of Daily Returns
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0

    # Return only new features
    return np.array([ma_5, ema_12, momentum, rsi, volatility])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Calculate relative thresholds based on historical std
    risk_threshold = 0.7  # Example threshold for high risk
    trend_threshold = 0.3  # Example threshold for trend direction

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold:
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return float(np.clip(reward, -100, 100))