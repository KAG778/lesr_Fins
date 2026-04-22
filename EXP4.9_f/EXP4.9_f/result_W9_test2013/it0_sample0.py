import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    closing_prices = s[::6]  # Extracting the closing prices
    volumes = s[4::6]        # Extracting the trading volumes

    # Feature 1: Price Change (percentage change from previous day)
    price_change = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices * 100
    # We take the difference and prepend the first closing price to maintain the shape

    # Feature 2: 5-day Moving Average of Closing Prices
    moving_average = np.convolve(closing_prices, np.ones(5)/5, mode='valid')
    # We will prepend NaN for the first 4 days to maintain the shape
    moving_average = np.concatenate((np.full(4, np.nan), moving_average))

    # Feature 3: Volume Change (percentage change from previous day)
    volume_change = np.diff(volumes, prepend=volumes[0]) / volumes * 100
    # Similarly, handle volume changes

    # Compile features into a single array while handling NaN values
    features = [
        price_change,
        moving_average,
        volume_change
    ]
    
    # Flatten the features and remove NaNs
    features = np.concatenate(features)
    features = np.nan_to_num(features)  # Convert NaNs to 0 for simplicity

    return features

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
        reward = -40  # STRONG NEGATIVE reward for BUY-aligned features
        # If we assume the action is BUY (action=0), we can return a strong penalty
        return reward
    elif risk_level > 0.4:
        reward = -20  # Moderate negative reward for BUY signals
        return reward

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive reward for BUY-aligned features in uptrend
        elif trend_direction < 0:
            reward += 10  # Positive reward for SELL-aligned features in downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming we have features indicating overbought/oversold
        # We would reward mean-reversion features here
        reward += 5  # Mild positive for mean-reversion strategies

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return reward