import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []

    # Extract closing prices and volumes
    closing_prices = s[::6]  # Every 6th element starting from index 0
    volumes = s[4::6]        # Every 6th element starting from index 4

    # Feature 1: Price Momentum (latest closing - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Volatility (Standard Deviation of the last 20 closing prices)
    volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0

    # Feature 3: Average Volume (last 20 days)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0

    # Feature 4: Relative Strength Index (RSI) - 14-day (using closing prices)
    if len(closing_prices) < 14:
        rsi = 0.0  # Handle edge case when we don't have enough data
    else:
        delta = np.diff(closing_prices[-14:])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain)
        avg_loss = np.mean(loss)
        
        # Avoid division by zero
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

    # Feature 5: Daily Return (percentage change)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else [0]
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0

    # Combine features
    features.extend([price_momentum, volatility, avg_volume, rsi, avg_daily_return])
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
        reward += -50  # Strong negative reward for BUY-aligned features
        reward += 10    # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if risk_level < 0.4:
        if abs(trend_direction) > 0.3:
            reward += 20 if trend_direction > 0 else 10  # Higher reward for alignment with upward trends
            reward += 10 if trend_direction < 0 else 5   # Positive reward for alignment with downward trends

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 15  # Example positive reward for mean reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds