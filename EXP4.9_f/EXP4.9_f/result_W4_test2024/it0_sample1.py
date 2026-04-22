import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extracting the necessary OHLCV data
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    opening_prices = s[1:120:6]  # Opening prices for 20 days
    high_prices = s[2:120:6]     # High prices for 20 days
    low_prices = s[3:120:6]      # Low prices for 20 days
    volumes = s[4:120:6]         # Trading volumes for 20 days

    # Feature 1: Price Change Percentage (last day closing to previous closing)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Daily Volume Change (percentage change in volume)
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 and np.all(volumes[:-1] != 0) else 0

    # Feature 3: Bollinger Band Upper and Lower (using last 20 days closing prices)
    moving_avg = np.mean(closing_prices)
    std_dev = np.std(closing_prices)
    upper_bb = moving_avg + (2 * std_dev)
    lower_bb = moving_avg - (2 * std_dev)

    # Feature 4: Last Closing Price Relative to Bollinger Bands
    price_relative_to_bb = 0
    if upper_bb != lower_bb:  # Avoid division by zero
        price_relative_to_bb = (closing_prices[-1] - lower_bb) / (upper_bb - lower_bb)

    # Return the computed features
    features = [price_change_pct, avg_volume_change, price_relative_to_bb]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 40  # STRONG NEGATIVE reward for BUY-aligned features
        return reward
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 15  # Positive reward for upward features
        else:
            reward += 15  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)