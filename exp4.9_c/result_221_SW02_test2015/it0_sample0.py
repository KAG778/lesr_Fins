import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Closing prices for the last two days
    closing_prices = s[0::6][:20]  # Extracting closing prices
    recent_closing_price = closing_prices[-1]
    previous_closing_price = closing_prices[-2] if len(closing_prices) > 1 else recent_closing_price

    # Feature 1: Price Momentum
    price_momentum = recent_closing_price - previous_closing_price
    features.append(price_momentum)

    # Feature 2: Average Volume
    volumes = s[4::6][:20]  # Extracting volumes
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    features.append(average_volume)

    # Feature 3: Daily Price Range
    high_prices = s[2::6][:20]  # Extracting high prices
    low_prices = s[3::6][:20]   # Extracting low prices
    daily_ranges = high_prices - low_prices
    recent_daily_range = daily_ranges[-1] if len(daily_ranges) > 0 else 0
    features.append(recent_daily_range)

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
        # Strong negative for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Strong negative reward
    elif risk_level > 0.4:
        # Moderate negative for BUY signals
        reward -= 20  # Moderate negative reward
    
    # If risk is low, we process further
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                reward += 10  # Positive reward for bullish features
            elif trend_direction < -0.3:
                reward += 10  # Positive reward for bearish features

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        if abs(trend_direction) < 0.3:
            # Reward mean-reversion features (example conditions)
            if enhanced_s[123] < -0.1:  # Assuming feature[0] indicates oversold
                reward += 10  # Reward for oversold condition
            elif enhanced_s[123] > 0.1:  # Assuming feature[1] indicates overbought
                reward += 10  # Reward for overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds