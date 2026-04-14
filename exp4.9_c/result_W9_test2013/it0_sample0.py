import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Feature 1: Price Change Percentage over the last 20 days
    try:
        price_change_pct = ((s[114] - s[0]) / s[0]) * 100  # (Recent closing - oldest closing) / oldest closing * 100
    except ZeroDivisionError:
        price_change_pct = 0.0  # Handle division by zero
    features.append(price_change_pct)

    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(s[4::6])  # Trading volume is every 6th item starting from index 4
    features.append(avg_volume)

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    closing_prices = s[0::6]  # Extract closing prices
    diff = np.diff(closing_prices)  # Daily price changes
    gain = np.where(diff > 0, diff, 0).mean()  # Average gain
    loss = np.where(diff < 0, -diff, 0).mean()  # Average loss

    try:
        rs = gain / loss  # Relative strength
        rsi = 100 - (100 / (1 + rs))  # RSI formula
    except ZeroDivisionError:
        rsi = 0.0  # Handle division by zero
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
        # STRONG NEGATIVE reward for BUY-aligned features
        reward += -40  # Example of strong negative reward for Buy
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example of moderate negative reward for Buy

    if risk_level > 0.7:
        reward += 5  # MILD POSITIVE reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for upward trend
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features
    
    # Penalize breakout-chasing features (not defined but could be included based on additional state)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% (uncertain market)

    # Ensure reward is bounded between [-100, 100]
    return float(np.clip(reward, -100, 100))