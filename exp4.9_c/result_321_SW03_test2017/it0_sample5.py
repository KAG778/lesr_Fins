import numpy as np

def revise_state(s):
    # s: 120d raw state
    # We will compute three features:
    # 1. Price Relative Strength (Current Price vs Moving Average)
    # 2. Volume Change Rate (Current Volume vs Previous Volume)
    # 3. Price Momentum (Closing Price Change Rate)
    
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]  # Extract trading volumes
    days = len(closing_prices)
    
    # Feature 1: Price Relative Strength
    moving_average = np.mean(closing_prices[-5:])  # 5-day moving average
    price_relative_strength = closing_prices[-1] / moving_average - 1  # Relative to the moving average
    features = [price_relative_strength]
    
    # Feature 2: Volume Change Rate
    volume_change_rate = 0
    if days > 1:
        volume_change_rate = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    features.append(volume_change_rate)
    
    # Feature 3: Price Momentum (Closing Price Change Rate)
    price_momentum = 0
    if days > 1:
        price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    features.append(price_momentum)
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = our computed features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Extracting features
    price_relative_strength = enhanced_state[123][0]
    volume_change_rate = enhanced_state[123][1]
    price_momentum = enhanced_state[123][2]
    
    # Initialize reward
    reward = 0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if price_relative_strength > 0:  # BUY-aligned features
            reward = np.random.uniform(-50, -30)  # Strong negative reward
        else:  # SELL-aligned features
            reward = np.random.uniform(5, 10)  # Mild positive reward
    elif risk_level > 0.4:
        if price_relative_strength > 0:  # BUY signals
            reward = np.random.uniform(-20, -10)  # Moderate negative reward
    
    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and price_relative_strength > 0:  # Strong upward signal
            reward = np.random.uniform(10, 30)  # Positive reward for following trend
        elif trend_direction < -0.3 and price_relative_strength < 0:  # Strong downward signal
            reward = np.random.uniform(10, 30)  # Positive reward for correct bearish bet
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if price_relative_strength < 0:  # Oversold condition
            reward = np.random.uniform(5, 15)  # Reward for potential buy
        elif price_relative_strength > 0:  # Overbought condition
            reward = np.random.uniform(-15, -5)  # Penalize breakout-chasing
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude
    
    return float(reward)