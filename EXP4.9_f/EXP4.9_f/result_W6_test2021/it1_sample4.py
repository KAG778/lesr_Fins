import numpy as np

def revise_state(s):
    closing_prices = s[0::6]
    volumes = s[4::6]
    
    # Feature 1: Price Momentum (last day closing price - average of last N days closing prices)
    N = 5  # Lookback for average
    if len(closing_prices) >= N:
        momentum = closing_prices[-1] - np.mean(closing_prices[-N:])
    else:
        momentum = 0.0  

    # Feature 2: Volatility (standard deviation of closing prices over the last N days)
    if len(closing_prices) >= N:
        volatility = np.std(closing_prices[-N:])
    else:
        volatility = 0.0  
    
    # Feature 3: Volume Change (percentage change from the previous day)
    if len(volumes) > 1:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0
    else:
        volume_change = 0.0  
    
    # Feature 4: Average True Range (ATR) for assessing market volatility
    if len(closing_prices) >= N:
        high_prices = s[2::6]
        low_prices = s[3::6]
        tr = np.maximum(high_prices[-N:] - low_prices[-N:], 
                        np.maximum(np.abs(high_prices[-N:] - closing_prices[-N:]), 
                                   np.abs(low_prices[-N:] - closing_prices[-N:])))
        atr = np.mean(tr)
    else:
        atr = 0.0  
    
    features = [momentum, volatility, volume_change, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(enhanced_s[:120])  # Estimate volatility from historical data
    high_risk_threshold = 0.7 * historical_volatility
    low_risk_threshold = 0.4 * historical_volatility
    
    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1: RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        if features[0] > 0:  # Positive momentum indicates a buy signal
            reward = -60  # Strong negative reward for BUY-aligned features
        else:
            reward = 15  # Mild positive reward for SELL-aligned features
    elif risk_level > low_risk_threshold:
        if features[0] > 0:  # Positive momentum
            reward = -30  # Moderate negative reward for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and positive momentum
            reward = 25  # Strong positive reward for upward alignment
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and negative momentum
            reward = 25  # Strong positive reward for downward alignment

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition indicates a buy signal
            reward = 20  # Reward for mean-reversion features
        else:
            reward = -15  # Penalize for chasing breakouts

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified bounds