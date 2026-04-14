import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes
    
    # Calculate returns
    returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Feature 1: Average True Range (ATR)
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR
    
    # Feature 2: Bollinger Bands (Upper and Lower)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1
    upper_band = moving_average + (2 * std_dev)
    lower_band = moving_average - (2 * std_dev)
    
    # Current price relative to Bollinger Bands
    distance_to_upper = (closing_prices[-1] - upper_band) / (upper_band if upper_band != 0 else 1)
    distance_to_lower = (closing_prices[-1] - lower_band) / (lower_band if lower_band != 0 else 1)
    
    # Feature 3: Rate of Change (Momentum)
    if len(closing_prices) >= 14:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15]  # 14-period ROC
    else:
        roc = 0
    
    features = [atr, distance_to_upper, distance_to_lower, roc]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    atr = features[0]
    distance_to_upper = features[1]
    distance_to_lower = features[2]
    roc = features[3]

    reward = 0.0

    # Calculate historical thresholds
    risk_threshold = 0.5 + (np.std([risk_level]) * 0.5)  # Relative based on historical std
    trend_threshold = 0.3 + (np.std([trend_direction]) * 0.2)  # Relative based on historical std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        # Strong negative reward for BUY-aligned features
        if roc > 0:  # Indicative of a BUY
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        else:
            reward += np.random.uniform(5, 10)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold and roc > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -trend_threshold and roc < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if distance_to_upper > 0:  # Overbought condition
            reward += np.random.uniform(10, 20)
        elif distance_to_lower < 0:  # Oversold condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]