import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices for the last 20 days
    opening_prices = s[1::6]  # Opening prices for the last 20 days
    high_prices = s[2::6]     # High prices for the last 20 days
    low_prices = s[3::6]      # Low prices for the last 20 days
    volumes = s[4::6]         # Trading volumes for the last 20 days
    
    # Feature 1: Price Change Rate (PCHR)
    price_change_rate = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Daily Volume (ADV)
    average_daily_volume = np.mean(volumes)

    # Feature 3: Volatility (standard deviation of returns over the last 5 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0  # Standard deviation of the last 5 returns

    # Return only the computed features
    return np.array([price_change_rate, average_daily_volume, volatility])

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
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        # Assuming SELL-aligned features are present, we can give a positive reward
        reward += 5  # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20  # positive reward for upward features (buying in an uptrend)
        elif trend_direction < -0.3:
            reward += 20  # positive reward for downward features (selling in a downtrend)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features (oversold→buy, overbought→sell)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% (uncertain market)

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward