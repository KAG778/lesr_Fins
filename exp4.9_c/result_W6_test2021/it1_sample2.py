import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: 5-Day Price Momentum (percentage change)
    try:
        price_momentum = (s[114] - s[109]) / s[109]  # (Close day 19 - Close day 14) / Close day 14
    except ZeroDivisionError:
        price_momentum = 0.0
    features.append(price_momentum)
    
    # Feature 2: 14-Day Average True Range (ATR) for volatility measurement
    def calculate_atr(highs, lows, closes, period=14):
        tr = np.maximum(highs[1:] - lows[1:], 
                        np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                   np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else 0
        return atr

    highs = s[2:120:6]  # Extract high prices
    lows = s[3:120:6]   # Extract low prices
    closes = s[0:120:6] # Extract closing prices
    atr = calculate_atr(highs, lows, closes)
    features.append(atr)

    # Feature 3: Z-score of the last 14 closing prices for mean reversion
    closing_prices = closes[-14:]  # Last 14 closing prices
    mean_price = np.mean(closing_prices)
    std_price = np.std(closing_prices)
    z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Calculate relative thresholds for risk
    risk_threshold_high = 0.7
    risk_threshold_mid = 0.4

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY signals
        reward += 10   # Mild positive reward for SELL signals
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= risk_threshold_mid:
        if trend_direction > 0:
            reward += 20  # Reward for buying in an uptrend
        elif trend_direction < 0:
            reward += 20  # Reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Strong reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)