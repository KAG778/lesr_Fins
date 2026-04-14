import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes
    
    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6]  # Most recent price - price 5 days ago
    features.append(price_momentum)
    
    # Feature 2: Average Volume (last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 0
    features.append(avg_volume)
    
    # Feature 3: Price Range (max high - min low over the last 20 days)
    price_range = np.max(s[2::6]) - np.min(s[3::6])  # max high - min low
    features.append(price_range)

    # Feature 4: Overbought/Oversold Indicator (RSI)
    delta = np.diff(closing_prices)
    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 5: Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0
    features.append(volatility)
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Calculate thresholds based on historical statistics
    historical_std = np.std(features) if features.size > 0 else 1
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        if features[3] < 30:  # RSI indicates oversold (buy)
            reward += 10  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)

    elif risk_level > medium_risk_threshold:
        reward += -20  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < medium_risk_threshold:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += 20
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < medium_risk_threshold:
        if features[3] < 30:  # RSI indicates oversold (buy)
            reward += 10  # Encourage buying
        elif features[3] > 70:  # RSI indicates overbought (sell)
            reward += 10  # Encourage selling
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]