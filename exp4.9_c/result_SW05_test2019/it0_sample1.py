import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    volumes = s[4:120:6]          # Extracting trading volumes
    
    # 1. Price Momentum: Current close - Close 5 days ago (with edge handling)
    momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0
    
    # 2. Volume Change: Percentage change in volume from day 1 to day 2
    volume_change = (volumes[0] - volumes[1]) / volumes[1] if volumes[1] != 0 else 0
    
    # 3. Moving Average Divergence (5-day MA vs 20-day MA)
    short_ma = np.mean(closing_prices[:5])
    long_ma = np.mean(closing_prices)
    ma_divergence = short_ma - long_ma
    
    features = [momentum, volume_change, ma_divergence]
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
    reward = 0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strongly negative for BUY-aligned features, mildly positive for SELL-aligned features
        reward = -40  # Strong negative reward for buying in high risk
    elif risk_level > 0.4:
        reward = -20  # Moderate negative for BUY signals in elevated risk
    
    # If risk is lower, check other priorities
    if risk_level <= 0.4:
        features = enhanced_s[123:]
        
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:  # Uptrend
                if features[0] > 0:  # Positive momentum
                    reward += 15  # Reward for correct long position
            else:  # Downtrend
                if features[0] < 0:  # Negative momentum
                    reward += 15  # Reward for correct short position
        
        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) <= 0.3:
            if risk_level < 0.3:  # Safe environment
                if features[0] < 0:  # Oversold condition
                    reward += 10  # Reward for buying in oversold
                elif features[0] > 0:  # Overbought condition
                    reward += 10  # Reward for selling in overbought
        
        # Priority 4 — HIGH VOLATILITY
        if volatility_level > 0.6:
            reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)