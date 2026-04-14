import numpy as np

def revise_state(s):
    features = []
    
    # Closing prices for the last 20 days
    closing_prices = s[0::6]
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Momentum (Rate of Change over the last 20 days)
    if len(closing_prices) >= 20 and closing_prices[19] != 0:
        momentum = (closing_prices[0] - closing_prices[19]) / closing_prices[19]
    else:
        momentum = 0
    features.append(momentum)
    
    # Feature 2: Average Volume Change (current volume vs. 20-day average)
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    if average_volume != 0:
        volume_change = (volumes[0] - average_volume) / average_volume
    else:
        volume_change = 0
    features.append(volume_change)
    
    # Feature 3: Bollinger Band Width (Volatility measure)
    if len(closing_prices) >= 20:
        std_dev = np.std(closing_prices[-20:])
        mean_price = np.mean(closing_prices[-20:])
        if mean_price != 0:
            bollinger_band_width = std_dev / mean_price
        else:
            bollinger_band_width = 0
    else:
        bollinger_band_width = 0
    features.append(bollinger_band_width)
    
    # Feature 4: Average True Range (ATR) for volatility measurement
    highs = s[2::6]
    lows = s[3::6]
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closing_prices[:-1]), 
                               np.abs(lows[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Assuming momentum is a BUY signal
            reward -= np.random.uniform(40, 60)  # Adjusted to be more severe
        # Mild positive reward for SELL-aligned features
        elif features[0] < 0:  # Assuming negative momentum is a SELL signal
            reward += np.random.uniform(10, 20)
    
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Assuming momentum is a BUY signal
            reward -= np.random.uniform(20, 40)  # Adjusted for severity
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            if features[0] > 0:  # Positive momentum
                reward += np.random.uniform(10, 30)  # Reward momentum alignment
        elif trend_direction < -0.3:  # Downtrend
            if features[0] < 0:  # Negative momentum
                reward += np.random.uniform(10, 30)  # Reward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += np.random.uniform(10, 20)  # Reward for buying
        elif features[0] > 0:  # Overbought condition
            reward -= np.random.uniform(10, 20)  # Penalize buying

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds