import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Bollinger Bands - upper and lower bands (20-day moving average +/- 2*std)
    closing_prices = s[0::6]  # Extracting closing prices
    moving_avg_20 = np.mean(closing_prices[-20:]) if len(closing_prices[-20:]) > 0 else 0
    std_dev_20 = np.std(closing_prices[-20:]) if len(closing_prices[-20:]) > 0 else 0
    upper_band = moving_avg_20 + (2 * std_dev_20)
    lower_band = moving_avg_20 - (2 * std_dev_20)
    features.extend([upper_band, lower_band])
    
    # Feature 2: Average True Range (ATR) - measures market volatility
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) > 0 else 0
    features.append(atr)

    # Feature 3: Weighted Moving Average (WMA) - gives more weight to recent prices
    weights = np.arange(1, 21)  # Weights for the last 20 days
    wma = np.sum(weights * closing_prices[-20:]) / np.sum(weights) if len(closing_prices[-20:]) >= 20 else 0
    features.append(wma)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    upper_band = features[0]
    lower_band = features[1]
    atr = features[2]
    wma = features[3]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50  # Strong negative reward for BUY-aligned features
        if wma < lower_band:  # Might consider selling if price is below lower band
            reward += 10  # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and wma > upper_band:  # Uptrend and price above upper band
            reward += 20  # Positive reward for strong upward features
        elif trend_direction < -0.3 and wma < lower_band:  # Downtrend and price below lower band
            reward += 20  # Positive reward for strong downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if wma < lower_band:  # Oversold condition
            reward += 20  # Encourage buying
        elif wma > upper_band:  # Overbought condition
            reward += 20  # Encourage selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds