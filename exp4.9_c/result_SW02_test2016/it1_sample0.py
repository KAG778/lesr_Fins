import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    volumes = s[4::6]  # Every 6th element starting from index 4
    
    # Edge case handling
    if len(closing_prices) < 2 or len(volumes) < 2:
        return np.zeros(4)  # Return zeros if there are not enough days of data

    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2]

    # Feature 2: Price Change Percentage ((close - open) / open) for the last day
    opening_price = s[1::6][-1]  # Opening price of the last day
    price_change_percentage = ((closing_prices[-1] - opening_price) / opening_price) * 100 if opening_price != 0 else 0

    # Feature 3: Volume Change (recent volume - previous volume)
    volume_change = volumes[-1] - volumes[-2]

    # Feature 4: Average True Range (ATR) for volatility measurement
    # Assuming 's' includes high and low prices as well (index 2 and 3 for this example)
    high_prices = s[2::6]
    low_prices = s[3::6]
    
    if len(high_prices) > 1 and len(low_prices) > 1:
        tr = np.maximum(high_prices[-1] - low_prices[-1], 
                        np.maximum(np.abs(high_prices[-1] - closing_prices[-2]), 
                                   np.abs(low_prices[-1] - closing_prices[-2])))
        atr = np.mean(tr[-14:])  # 14-day ATR
    else:
        atr = 0  # Default to 0 if not enough data

    features = [price_momentum, price_change_percentage, volume_change, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical std as a relative threshold for dynamic risk assessment
    historical_std = np.std(enhanced_s[:120])  # Assuming we calculate it from the raw state

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(50, 70)  # Strong negative for BUY-aligned features
        reward += 10  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(20, 30)  # Positive reward for uptrend features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(20, 30)  # Positive reward for downtrend features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(15, 25)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain the reward to [-100, 100]
    return np.clip(reward, -100, 100)