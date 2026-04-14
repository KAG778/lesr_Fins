import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []
    
    # Extracting the closing prices and volumes
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    volumes = s[4::6]         # Every 6th element starting from index 4

    # Feature 1: 5-day Price Change Percentage
    if len(closing_prices) >= 6:
        price_change_5d = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100
    else:
        price_change_5d = 0
    features.append(price_change_5d)
    
    # Feature 2: 14-day Relative Strength Index (RSI)
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Calculate price changes
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / (loss + 1e-8)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI
    features.append(rsi)

    # Feature 3: 20-day Volatility (Standard Deviation of Daily Returns)
    if len(closing_prices) > 1:
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]
        volatility = np.std(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    else:
        volatility = 0
    features.append(volatility)

    # Feature 4: 5-day Moving Average vs 20-day Moving Average
    if len(closing_prices) >= 20:
        short_ma = np.mean(closing_prices[-5:])
        long_ma = np.mean(closing_prices[-20:])
        ma_diff = short_ma - long_ma
    else:
        ma_diff = 0
    features.append(ma_diff)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Define relative thresholds for risk management
    historical_volatility = np.std(features[2])  # Using the volatility feature
    high_risk_threshold = 0.7 * historical_volatility
    medium_risk_threshold = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Positive price change indicates BUY
            reward = -40  # Strong negative reward
        else:  # Negative price change indicates SELL
            reward = 10  # Mild positive reward for selling
    elif risk_level > medium_risk_threshold:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward = -20  # Negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive price change
            reward += 20  # Positive reward for following the trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative price change
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition
            reward += 15  # Reward for buying
        elif features[1] > 70:  # Overbought condition
            reward -= 15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level <= medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clipping the reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward