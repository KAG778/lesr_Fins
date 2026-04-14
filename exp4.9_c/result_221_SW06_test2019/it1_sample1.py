import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]
    
    # Feature 1: 30-day Exponential Moving Average (EMA)
    ema_period = 30
    ema = np.mean(closing_prices[-ema_period:]) if len(closing_prices) >= ema_period else 0
    features.append(ema)

    # Feature 2: Average True Range (ATR) for volatility
    true_ranges = np.maximum(s[2:120:6] - s[3:120:6], 
                             np.maximum(abs(s[2:120:6] - s[1:120:6]), 
                                        abs(s[3:120:6] - s[1:120:6])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0
    features.append(atr)

    # Feature 3: Rate of Change (ROC) - Momentum
    roc_period = 12
    if len(closing_prices) > roc_period:
        roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]
    else:
        roc = 0
    features.append(roc)

    # Feature 4: Market Breadth - Difference between advancing and declining issues
    # Assuming s[5:120:6] is a hypothetical array of advancing issues and s[6:120:6] is declining issues
    advancing_issues = s[5:120:6]  # Placeholder for advancing issues
    declining_issues = s[6:120:6]  # Placeholder for declining issues
    breadth = np.sum(advancing_issues) - np.sum(declining_issues)
    features.append(breadth)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(features)  # Standard deviation of the features as a proxy for volatility
    high_risk_threshold = historical_volatility * 1.5  # Example of a relative threshold for high risk
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[2] > 0:  # Assuming feature[2] is the ROC indicating bullish momentum
            reward -= np.random.uniform(50, 100)  # Strong penalty for buying in a high-risk environment
        else:  # Assuming feature indicates a SELL signal
            reward += np.random.uniform(10, 20)  # Mild positive reward for selling

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[2] > 0:  # Uptrend with positive momentum
            reward += 20  
        elif trend_direction < 0 and features[2] < 0:  # Downtrend with negative momentum
            reward += 20  

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0:  # Oversold condition
            reward += 10  # Encourage buying during mean reversion
        elif features[2] > 0:  # Overbought condition
            reward -= 10  # Discourage buying during overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds