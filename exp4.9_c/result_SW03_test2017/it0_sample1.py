import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    closing_prices = s[0:120:6]  # Extract closing prices
    opening_prices = s[1:120:6]  # Extract opening prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Price momentum (Rate of Change)
    price_momentum = closing_prices[-1] / closing_prices[-2] - 1 if closing_prices[-2] != 0 else 0
    
    # Feature 2: Price range (High - Low)
    price_range = high_prices[-1] - low_prices[-1]

    # Feature 3: Average trading volume over the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # Feature 4: Price change from open to close (percentage)
    price_change = (closing_prices[-1] - opening_prices[-1]) / opening_prices[-1] if opening_prices[-1] != 0 else 0
    
    features = [price_momentum, price_range, avg_volume, price_change]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Negative reward for risky buying
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 15  # This can be adjusted based on the features for buy signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        features = enhanced_s[123:]  # Retrieve computed features
        if trend_direction > 0.3:
            # Positive reward for upward features
            reward += 10 * features[0]  # Assuming first feature indicates upward momentum
        elif trend_direction < -0.3:
            # Positive reward for downward features
            reward += 10 * (1 - features[0])  # Assuming first feature indicates downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        features = enhanced_s[123:]
        # Reward mean-reversion features
        reward += 10 * (1 - features[0])  # Assuming first feature indicates mean-reversion potential

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Return reward within the specified range