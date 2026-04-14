import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Price change percentage over the last 20 days
    price_change = (closing_prices[-1] - closing_prices[0]) / closing_prices[0] if closing_prices[0] != 0 else 0

    # Feature 2: Average volume over the last 20 days
    avg_volume = np.mean(volumes)

    # Feature 3: Bollinger Bands (20-day moving average and standard deviation)
    moving_avg = np.mean(closing_prices)
    moving_std = np.std(closing_prices)
    lower_band = moving_avg - (2 * moving_std)
    upper_band = moving_avg + (2 * moving_std)
    
    # Feature indicating if the current closing price is above or below the bands
    price_above_upper_band = 1.0 if closing_prices[-1] > upper_band else 0.0
    price_below_lower_band = 1.0 if closing_prices[-1] < lower_band else 0.0
    
    # Return the computed features
    features = [price_change, avg_volume, price_above_upper_band, price_below_lower_band]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0  # Initialize reward

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative reward for BUY-aligned features
        return reward  # Immediate return to prioritize risk management
    elif risk_level > 0.4:
        reward += -15  # Moderate negative reward for BUY signals

    # Extract features
    features = enhanced_state[123:]

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Reward for correct trend direction alignment
        if trend_direction > 0 and features[0] > 0:  # Upward momentum
            reward += 20  # Positive reward for strong upward features
        elif trend_direction < 0 and features[3] > 0:  # Downward momentum
            reward += 20  # Positive reward for strong downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] > 0:  # Price above upper band
            reward += -10  # Penalize for chasing breakouts
        if features[3] > 0:  # Price below lower band
            reward += 10  # Reward for mean-reverting behavior

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward