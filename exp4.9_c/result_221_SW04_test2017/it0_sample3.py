import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Calculate features based on the last 20 days of prices
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)

    # Feature 1: 5-day moving average
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = np.nan  # Handle edge case, not enough data

    # Feature 2: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.nan
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.nan
    rs = avg_gain / avg_loss if avg_loss > 0 else np.nan  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan

    # Feature 3: Historical Volatility (30 days)
    returns = np.diff(np.log(closing_prices))
    historical_volatility = np.std(returns[-30:]) if len(returns) >= 30 else np.nan  # Handle edge case

    features = [moving_average, rsi, historical_volatility]
    
    # Filter out NaN values
    return np.nan_to_num(np.array(features))

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
        # Strong negative reward for BUY signals
        reward -= np.random.uniform(30, 50)  # Strong negative for risky buy
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(5, 15)  # Moderate negative for risky buy

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for bullish trends
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for bearish trends

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += 5  # Mild positive for mean-reversion strategies

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]