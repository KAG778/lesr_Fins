import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6][:20]  # Extract last 20 closing prices
    volumes = s[4::6][:20]  # Extract last 20 trading volumes

    # Feature 1: Price Momentum (current closing price - closing price 10 days ago)
    momentum = closing_prices[-1] - closing_prices[-11] if len(closing_prices) > 10 else 0

    # Feature 2: Average Trading Volume over the last 10 days
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)

    # Feature 3: Historical Volatility (Standard Deviation of returns over the last 20 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(returns) if len(returns) >= 20 else 0

    # Feature 4: Average True Range (ATR) to capture market volatility
    high_prices = s[2::6][:20]  # Extract high prices
    low_prices = s[3::6][:20]   # Extract low prices
    atr = np.mean(np.maximum(high_prices[1:] - low_prices[1:], 
                              np.abs(high_prices[1:] - closing_prices[:-1]),
                              np.abs(low_prices[1:] - closing_prices[:-1])))

    features = [momentum, avg_volume, historical_volatility, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(20, 50)  # Strong negative reward for BUY
        reward += np.random.uniform(5, 15)    # Mild positive reward for SELL
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)  # Mild negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += np.random.uniform(10, 25)  # Positive reward for upward momentum
        elif trend_direction < 0:
            reward += np.random.uniform(10, 25)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features
        reward -= np.random.uniform(5, 10)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds