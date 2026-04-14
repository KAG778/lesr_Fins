import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes
    n = len(closing_prices)

    features = []

    # Feature 1: Average True Range (ATR)
    if n > 1:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                   abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr) if len(tr) > 0 else 0
        features.append(atr)
    else:
        features.append(0)

    # Feature 2: Bollinger Bands (Upper and Lower Bands)
    if n > 20:
        moving_avg = np.mean(closing_prices[-20:])
        moving_std = np.std(closing_prices[-20:])
        upper_band = moving_avg + (2 * moving_std)
        lower_band = moving_avg - (2 * moving_std)
        features.append(upper_band - lower_band)  # Width of the band
    else:
        features.append(0)

    # Feature 3: Momentum Indicator (Rate of Change)
    if n > 1:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
        features.append(momentum)
    else:
        features.append(0)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0
    
    # Priority 1: RISK MANAGEMENT
    risk_threshold = 0.7  # Example value, may be dynamically set using historical std
    if risk_level > risk_threshold:
        if features[0] > 0:  # Assume positive feature indicates BUY
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
        return np.clip(reward, -100, 100)
    
    # Priority 2: TREND FOLLOWING
    trend_threshold = 0.3  # Example value, may be dynamically set using historical std
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:  # Uptrend
            reward += features[2] * 20  # Reward based on momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += -features[2] * 20  # Reward for correct bearish momentum

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[1] < 0:  # Assuming Bollinger Band indicates oversold
            reward += 10  # Reward for buying in oversold condition
        elif features[1] > 0:  # Assuming Bollinger Band indicates overbought
            reward += 10  # Reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds