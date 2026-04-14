import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    num_days = len(closing_prices)

    # Feature 1: 20-day Standard Deviation of Closing Prices
    if num_days >= 20:
        price_std = np.std(closing_prices[-20:])
    else:
        price_std = 0

    # Feature 2: Average True Range (ATR) over the last 14 days
    if num_days >= 14:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[-14:] - low_prices[-14:], 
                        np.abs(high_prices[-14:] - closing_prices[-15:-1]), 
                        np.abs(low_prices[-14:] - closing_prices[-15:-1]))
        atr = np.mean(tr)
    else:
        atr = 0

    # Feature 3: 14-day Momentum (current price - price 14 days ago)
    momentum = closing_prices[-1] - closing_prices[-15] if num_days >= 15 else 0

    # Feature 4: Market Breadth (example placeholder logic)
    # In a real strategy, this would be calculated based on a broader index
    market_breadth = np.random.uniform(-1, 1)  # Placeholder for market breadth calculation

    # Compile features
    features = [price_std, atr, momentum, market_breadth]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong penalty for high-risk conditions
    elif risk_level > 0.4:
        reward -= 20  # Mild penalty for moderate-risk conditions

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 30 * np.sign(trend_direction)  # Positive reward for aligning with trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified range