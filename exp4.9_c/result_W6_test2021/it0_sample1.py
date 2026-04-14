import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []

    # Feature 1: Price momentum (percentage change over the last 5 days)
    try:
        price_change = (s[114] - s[108]) / s[108]  # (Close day 19 - Close day 14) / Close day 14
    except ZeroDivisionError:
        price_change = 0.0
    features.append(price_change)

    # Feature 2: Average volume over the last 5 days
    avg_volume = np.mean(s[4::6][:5])  # Average of the last 5 days' trading volume
    features.append(avg_volume)

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0).mean()
        loss = np.abs(np.where(delta < 0, delta, 0)).mean()
        rs = gain / loss if loss > 0 else 0
        return 100 - (100 / (1 + rs))

    closing_prices = s[0::6][:14]  # Last 14 closing prices
    rsi = calculate_rsi(closing_prices)
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
        reward -= np.random.uniform(30, 50)  # Random strong negative reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 15

    # If not heavily risky, proceed to evaluate trend and volatility
    if risk_level <= 0.4:

        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                reward += 20  # Reward for being in an upward trend
            elif trend_direction < -0.3:
                reward += 20  # Reward for being in a downward trend

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3 and risk_level < 0.3:
            # Reward mean-reversion features & penalize breakout chasing
            reward += 10  # Mean reversion reward

        # Priority 4 — HIGH VOLATILITY
        if volatility_level > 0.6:
            reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is in the range [-100, 100]