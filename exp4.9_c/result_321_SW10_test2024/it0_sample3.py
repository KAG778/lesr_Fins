import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    
    # Feature 1: Price Change (% Change from previous closing price)
    price_changes = np.diff(closing_prices) / closing_prices[:-1]
    price_change = np.mean(price_changes) if len(price_changes) > 0 else 0.0
    
    # Feature 2: Average True Range (ATR) over the last 20 days
    highs = s[2:120:6]
    lows = s[3:120:6]
    atr = np.mean(np.maximum(highs[1:] - lows[1:], highs[:-1] - closing_prices[:-1], closing_prices[:-1] - lows[:-1])) if len(highs) > 1 else 0.0
    
    # Feature 3: Volume Change (% Change from previous volume)
    volume_changes = np.diff(volumes) / volumes[:-1]
    volume_change = np.mean(volume_changes) if len(volume_changes) > 0 else 0.0
    
    # Return computed features as a numpy array
    return np.array([price_change, atr, volume_change])

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

    # Priority 1: Risk Management (deterministic, NO random calls)
    if risk_level > 0.7:
        reward -= 45.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 15.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[0] > 0:  # Positive price change
            reward += trend_direction * features[0] * 20.0  # Reward for following trend
        elif features[0] < 0:  # Negative price change
            reward += -trend_direction * features[0] * 20.0  # Reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 10.0  # Reward for considering a BUY
        elif features[0] > 0:  # Overbought condition
            reward += 10.0  # Reward for considering a SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))