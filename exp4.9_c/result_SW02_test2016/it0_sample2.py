import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extract the closing prices and volumes from the raw state
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    volumes = s[4::6]  # Every 6th element starting from index 4
    
    # Edge case handling
    if len(closing_prices) < 2 or len(volumes) < 2:
        return np.zeros(3)  # Return zeros if there are not enough days of data

    # Feature 1: Price Momentum (recent close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2]

    # Feature 2: Price Change Percentage ((close - open) / open) for the last day
    opening_price = s[1::6][-1]  # Opening price of the last day
    price_change_percentage = ((closing_prices[-1] - opening_price) / opening_price) * 100 if opening_price != 0 else 0

    # Feature 3: Volume Change (recent volume - previous volume)
    volume_change = volumes[-1] - volumes[-2]

    features = [price_momentum, price_change_percentage, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY signals
        if enhanced_state[123] > 0:  # Assuming positive feature indicates a buy signal
            reward = np.random.uniform(-50, -30)  # Strong negative reward
        else:
            reward = np.random.uniform(5, 10)  # Mild positive reward for sell signals
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if enhanced_state[123] > 0:  # Assuming positive feature indicates a buy signal
            reward = np.random.uniform(-10, -5)

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_state[123] > 0:  # Buy signal in uptrend
            reward += 20  # Positive reward for correct trend following
        elif trend_direction < -0.3 and enhanced_state[123] < 0:  # Sell signal in downtrend
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_state[123] > 0:  # Oversold signal
            reward += 15  # Positive reward for buying oversold
        elif enhanced_state[123] < 0:  # Overbought signal
            reward += 15  # Positive reward for selling overbought

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)