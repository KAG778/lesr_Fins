import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract volumes
    
    # 1. Price Momentum: Calculate the difference between the most recent closing price and the one from 10 days ago
    price_momentum = closing_prices[0] - closing_prices[10] if len(closing_prices) > 10 else 0
    
    # 2. Average Trading Volume: Calculate the average volume over the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    
    # 3. Price Range: Calculate the average price range (high - low) over the last 20 days
    price_range = np.mean(high_prices - low_prices) if len(high_prices) > 0 else 0

    features = [price_momentum, avg_volume, price_range]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management (deterministic, NO random calls)
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= 40.0  # Strong negative for buying
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 10.0  # Moderate negative for buying

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if len(features) > 0:
            reward += trend_direction * features[0] * 10.0  # Use price momentum feature

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Mild positive for buying
        elif features[0] > 0:  # Overbought condition
            reward -= 5.0  # Mild negative for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))