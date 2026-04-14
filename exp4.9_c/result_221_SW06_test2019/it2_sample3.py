import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices
    closing_prices = s[0:120:6]
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    
    # Feature 1: Average True Range (ATR) for volatility
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    features.append(atr)

    # Feature 2: Z-score of the last closing price
    mean_price = np.mean(closing_prices[-20:])  # Mean of last 20 closing prices
    std_price = np.std(closing_prices[-20:])    # Std dev of last 20 closing prices
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    # Feature 3: Rate of Change (ROC) - Momentum
    roc_period = 10
    roc = (closing_prices[-1] - closing_prices[-(roc_period + 1)]) / closing_prices[-(roc_period + 1)] if len(closing_prices) > roc_period else 0
    features.append(roc)

    # Feature 4: 14-day Momentum Indicator
    momentum = np.sum(closing_prices[-14:] - closing_prices[-15:-1])  # Sum of the price changes over the last 14 days
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # The new features
    reward = 0.0

    # Calculate relative thresholds based on historical std
    historical_std = np.std(features)
    low_threshold = np.mean(features) - (1 * historical_std)
    high_threshold = np.mean(features) + (1 * historical_std)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[2] > 0:  # Assuming feature[2] is the ROC indicating bullish momentum
            reward -= np.random.uniform(30, 50)  # Strong penalty for buying in a high-risk environment
        else:  # Assuming feature indicates a SELL signal
            reward += np.random.uniform(5, 10)  # Mild positive reward for selling

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[2] > 0:  # Uptrend with positive momentum
            reward += 20  
        elif trend_direction < 0 and features[2] < 0:  # Downtrend with negative momentum
            reward += 20  

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < low_threshold:  # Oversold condition
            reward += 10  # Encourage buying during mean reversion
        elif features[1] > high_threshold:  # Overbought condition
            reward -= 10  # Discourage buying during overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds