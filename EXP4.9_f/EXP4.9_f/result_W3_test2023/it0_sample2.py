import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Feature 1: Price momentum (change in closing price over the last 5 days)
    closing_prices = s[::6][:20]  # Extracting the closing prices
    momentum = np.zeros(20)
    for i in range(1, len(closing_prices)):
        momentum[i] = (closing_prices[i] - closing_prices[i-1]) / (closing_prices[i-1] + 1e-10)  # Avoid division by zero
    
    # Feature 2: Average trading volume over the last 5 days
    trading_volumes = s[4::6][:20]  # Extracting the trading volumes
    avg_volume = np.mean(trading_volumes[-5:]) if len(trading_volumes) >= 5 else 0
    
    # Feature 3: Price range (high - low) as a measure of volatility
    high_prices = s[2::6][:20]  # Extracting the high prices
    low_prices = s[3::6][:20]   # Extracting the low prices
    price_range = high_prices - low_prices
    
    # Combine the features into a single array
    features = [momentum[-1], avg_volume, np.mean(price_range[-5:])]  # Use last momentum value, avg volume, and avg price range
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
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming features[0] indicates a bullish signal (momentum)
            return np.random.uniform(-50, -30)  # STRONG NEGATIVE reward for BUY-aligned features
        else:
            return np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        if features[0] > 0:  # Assuming features[0] indicates a bullish signal (momentum)
            reward += -10  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Bullish trend and bullish signal
            reward += 20  # Positive reward for correct bullish bet
        elif trend_direction < -0.3 and features[0] < 0:  # Bearish trend and bearish signal
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 15  # Reward buying
        else:
            reward += -15  # Penalize for breakout-chasing features (buying in a sideways market)
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return reward