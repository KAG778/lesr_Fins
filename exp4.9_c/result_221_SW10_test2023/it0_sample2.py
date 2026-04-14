import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices from the raw state
    volumes = s[4:120:6]          # Extract volumes from the raw state

    # Calculate Price Momentum
    momentum = closing_prices[-1] - closing_prices[-6]  # Current close - Close 5 days ago
    momentum_std = np.std(closing_prices[-5:])  # Standard deviation of last 5 closing prices
    price_momentum = momentum / momentum_std if momentum_std != 0 else 0

    # Calculate Volume Change
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0

    # Calculate RSI
    gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
    avg_gain = np.mean(gains[-14:])  # Average gain over the last 14 periods
    avg_loss = np.mean(losses[-14:])  # Average loss over the last 14 periods
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Return the computed features as a numpy array
    features = [price_momentum, volume_change, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive momentum indicates a buy signal
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # Assuming negative momentum indicates a sell signal
            reward += np.random.uniform(5, 10)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Positive momentum
            reward -= 10

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += 10
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += 10

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Assuming RSI < 30 is oversold
            reward += 10
        elif features[2] > 70:  # Assuming RSI > 70 is overbought
            reward += 10

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the bounds
    return max(-100, min(100, reward))