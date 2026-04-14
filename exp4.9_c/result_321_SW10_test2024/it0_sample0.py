import numpy as np

def revise_state(s):
    # s: 120d raw state
    n_days = 20
    closing_prices = s[0::6][:n_days]  # Extract closing prices
    opening_prices = s[1::6][:n_days]  # Extract opening prices
    volumes = s[4::6][:n_days]          # Extract trading volumes
    highs = s[2::6][:n_days]            # Extract high prices
    lows = s[3::6][:n_days]             # Extract low prices
    
    # Feature 1: Price Momentum
    price_momentum = (closing_prices[-1] - opening_prices[-1]) / (opening_prices[-1] + 1e-10)
    
    # Feature 2: Volume Change
    volume_change = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-10) if n_days > 1 else 0.0

    # Feature 3: Price Range normalized by closing price
    price_range = (highs[-1] - lows[-1]) / (closing_prices[-1] + 1e-10)

    # Return the computed features as a numpy array
    return np.array([price_momentum, volume_change, price_range])

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_state[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price momentum
            reward += 10.0 * features[0]
        elif features[0] < 0:  # Negative price momentum
            reward += 10.0 * -features[0]

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Reward for potential BUY
        elif features[0] > 0:  # Overbought condition
            reward += 5.0  # Reward for potential SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))