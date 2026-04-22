import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extracting the closing prices for calculation
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    opening_prices = s[1::6]  # Every 6th element starting from index 1
    
    # Feature 1: Price Momentum (Percentage Change)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    
    # Feature 2: Average Trading Volume (last 5 days)
    trading_volumes = s[4::6]  # Every 6th element starting from index 4 (trading volume)
    avg_volume = np.mean(trading_volumes[-5:]) if len(trading_volumes) >= 5 else 0
    
    # Feature 3: Price Range (High - Low) of the last day
    high_price = s[3::6][-1]  # Last high price
    low_price = s[4::6][-1]   # Last low price
    price_range = high_price - low_price
    
    # Return the features as a numpy array
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
    
    reward = 0.0
    features = enhanced_s[123:]
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        if features[0] > 0:  # if price momentum is positive (indicating a buy signal)
            reward = -40
        # MILD POSITIVE reward for SELL-aligned features
        elif features[0] < 0:  # if price momentum is negative (indicating a sell signal)
            reward = 10
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward = -20

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # uptrend and positive momentum
            reward = 15
        elif trend_direction < -0.3 and features[0] < 0:  # downtrend and negative momentum
            reward = 15

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        if features[0] < 0:  # oversold (indicating a buy signal)
            reward = 10
        elif features[0] > 0:  # overbought (indicating a sell signal)
            reward = -10

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)