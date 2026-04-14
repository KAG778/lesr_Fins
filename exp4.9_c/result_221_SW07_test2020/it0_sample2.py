import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]  # Extract low prices
    
    # Calculate daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Feature 1: Average Daily Return (over the last 19 days)
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
    
    # Feature 2: Price Relative to Opening Price (most recent day)
    price_rel_to_open = (closing_prices[-1] - opening_prices[-1]) / opening_prices[-1] if opening_prices[-1] != 0 else 0
    
    # Feature 3: High-Low Range (most recent day)
    high_low_range = high_prices[-1] - low_prices[-1] if high_prices[-1] > low_prices[-1] else 0
    
    features = [avg_daily_return, price_rel_to_open, high_low_range]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming feature[0] relates to positive returns
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # Assuming feature[0] relates to negative returns
            reward += np.random.uniform(5, 10)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= 10  # Example of a moderate negative reward
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Assuming feature[0] favors buying
                reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Assuming feature[0] favors selling
                reward += 10  # Positive reward for downward features
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        if features[0] < 0:  # Assuming feature[0] indicates oversold
            reward += 10  # Positive reward for buying
        elif features[0] > 0:  # Assuming feature[0] indicates overbought
            reward += 10  # Positive reward for selling
        # Penalize breakout-chasing features
        if features[1] > 0:  # Assuming feature[1] relates to breakout chasing
            reward -= 5  # Penalize breakout-chasing
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)