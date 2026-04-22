import numpy as np

def revise_state(s):
    closing_prices = s[::6][:20]  # Extract closing prices
    volumes = s[4::6][:20]          # Extract trading volumes

    # Feature 1: Price Momentum (change in closing price over the last 5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: 5-day Simple Moving Average (SMA) 
    sma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0

    # Feature 3: Average Volume Change (current volume - average volume over the last 5 days)
    avg_volume_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1e-10  # Avoid division by zero
    volume_change = (volumes[-1] - avg_volume_5) / avg_volume_5 if avg_volume_5 != 0 else 0

    # Feature 4: Relative Strength Index (RSI) over 14 days
    gains = np.maximum(0, np.diff(closing_prices[-14:]))  # Daily gains
    losses = np.abs(np.minimum(0, np.diff(closing_prices[-14:])))  # Daily losses
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10  # Avoid division by zero
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 5: Volatility as standard deviation of returns over the last 5 days
    returns = np.diff(closing_prices[-5:]) / closing_prices[-6:-1] if len(closing_prices) >= 6 else [0]
    volatility = np.std(returns)

    features = [price_momentum, sma_5, volume_change, rsi, volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Price momentum suggests a BUY
            reward = np.random.uniform(-50, -30)  # Strong negative for BUY-aligned features
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for SELL-aligned features
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if (trend_direction > 0 and features[0] > 0) or (trend_direction < 0 and features[0] < 0):
            reward += 20  # Positive reward for momentum alignment with trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward for mean-reversion
        if features[3] < 30:  # Oversold condition
            reward += 10  # Reward for potential buy
        elif features[3] > 70:  # Overbought condition
            reward += 10  # Reward for potential sell
        else:
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds
    return np.clip(reward, -100, 100)