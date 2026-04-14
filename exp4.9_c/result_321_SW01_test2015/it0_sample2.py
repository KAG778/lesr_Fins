import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract the closing prices
    volumes = s[4::6]  # Extract the trading volumes

    # Feature 1: Price Change Percentage
    price_changes = np.diff(closing_prices) / closing_prices[:-1] # percentage change
    price_change_percentage = np.pad(price_changes, (1, 0), 'constant', constant_values=np.nan)  # pad to match shapes

    # Feature 2: 5-Day Moving Average of Closing Prices
    moving_average = np.convolve(closing_prices, np.ones(5)/5, mode='valid')
    moving_average = np.pad(moving_average, (2, 0), 'constant', constant_values=np.nan)  # pad to match shapes

    # Feature 3: Volume Change Percentage
    volume_changes = np.diff(volumes) / volumes[:-1]  # percentage change
    volume_change_percentage = np.pad(volume_changes, (1, 0), 'constant', constant_values=np.nan)  # pad to match shapes

    features = [price_change_percentage[-1], moving_average[-1], volume_change_percentage[-1]]
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
        if enhanced_s[123] == 0:  # Assume this means BUY action
            reward = np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY
        else:
            reward = np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        if enhanced_s[123] == 0:  # Assume this means BUY action
            reward = np.random.uniform(-15, -5)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            if enhanced_s[123] == 0:  # Assume this means BUY action
                reward += 10  # Positive reward for correct bullish bet
        elif trend_direction < -0.3:
            if enhanced_s[123] == 1:  # Assume this means SELL action
                reward += 10  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming certain features indicate oversold/overbought conditions
        if enhanced_s[123] == 0:  # Assume this means BUY during oversold
            reward += 5  # Reward for mean-reversion BUY
        elif enhanced_s[123] == 1:  # Assume this means SELL during overbought
            reward += 5  # Reward for mean-reversion SELL

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)