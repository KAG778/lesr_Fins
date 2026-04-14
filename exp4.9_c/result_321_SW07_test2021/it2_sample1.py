import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Momentum (current close - close 5 days ago) normalized by current price
    price_momentum = (s[6*19] - s[6*14]) / s[6*19] if s[6*19] != 0 else 0
    features.append(price_momentum)
    
    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(s[4::6])  # Every 6th entry starting from index 4
    features.append(avg_volume)
    
    # Feature 3: Price Range (max high - min low) normalized by average price
    high_prices = s[2::6]
    low_prices = s[3::6]
    price_range = (np.max(high_prices) - np.min(low_prices)) / np.mean(s[0::6]) if np.mean(s[0::6]) != 0 else 0
    features.append(price_range)
    
    # Feature 4: Relative Strength Index (RSI) for the last 14 days
    closing_prices = s[0::6]
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean() if np.any(delta > 0) else 0
    loss = -np.where(delta < 0, delta, 0).mean() if np.any(delta < 0) else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)
    
    # Feature 5: Volatility (standard deviation of daily returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0
    features.append(volatility)
    
    # Feature 6: Z-score of the last 5 closing prices
    mean_price = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0
    std_price = np.std(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 1  # Avoid division by zero
    z_score = (closing_prices[-1] - mean_price) / std_price
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract new features
    reward = 0.0
    
    # Calculate dynamic thresholds based on historical std deviations
    historical_std = np.std(features) if features.size > 0 else 1
    risk_threshold_high = risk_level > 0.7
    risk_threshold_medium = risk_level > 0.4
    
    # Priority 1 — RISK MANAGEMENT
    if risk_threshold_high:
        reward += -50  # Strong negative for BUY-aligned features
        if features[0] < 0:  # Price momentum is negative, consider it a sell scenario
            reward += 10  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)

    elif risk_threshold_medium:
        reward += -20  # Moderate negative for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and not risk_threshold_high:
        if (trend_direction > 0 and features[0] > 0) or (trend_direction < 0 and features[0] < 0):
            reward += 20  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and not risk_threshold_high:
        if features[3] < 30:  # Oversold condition for RSI
            reward += 15  # Encourage buying
        elif features[3] > 70:  # Overbought condition for RSI
            reward += 15  # Encourage selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]