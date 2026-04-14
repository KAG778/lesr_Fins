import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    N = len(closing_prices)

    # Feature 1: 10-Day Moving Average (MA)
    ma = np.mean(closing_prices[-10:]) if N >= 10 else np.nan

    # Feature 2: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-14:]) if N >= 14 else np.nan
    avg_loss = np.mean(loss[-14:]) if N >= 14 else np.nan
    
    rs = avg_gain / avg_loss if avg_loss > 0 else np.nan  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan

    # Feature 3: Historical Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-10:]) if N >= 10 else np.nan

    # Feature 4: Volume Change Percentage
    if len(volumes) > 1:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]  # Percentage change in volume
    else:
        volume_change = np.nan

    # Feature 5: Sentiment Analysis (if available, e.g., via external API)
    # Assuming sentiment_score is a placeholder for an external sentiment analysis feature.
    sentiment_score = np.random.rand() * 2 - 1  # Randomly generate for illustration; replace with actual score

    features = [ma, rsi, volatility, volume_change, sentiment_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # STRONG NEGATIVE reward for BUY-aligned features
        return max(-100, min(100, reward))  # Early return as risk is high
    elif risk_level > 0.4:
        reward -= 15  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for upward features
        else:  # Downtrend
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return max(-100, min(100, reward))  # Ensure reward stays within bounds