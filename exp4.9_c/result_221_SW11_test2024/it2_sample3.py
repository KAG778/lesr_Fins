import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes
    n = len(closing_prices)

    features = []

    # Feature 1: Exponential Moving Average (EMA) - 20 days
    if n >= 20:
        ema_20 = np.mean(closing_prices[-20:])
    else:
        ema_20 = closing_prices[-1] if n > 0 else 0
    features.append(ema_20)

    # Feature 2: Average True Range (ATR) - 14 days
    if n >= 14:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                   abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])
    else:
        atr = 0
    features.append(atr)

    # Feature 3: Rate of Change (ROC) of closing prices
    if n > 1:
        roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    else:
        roc = 0
    features.append(roc)

    # Feature 4: Distance from Upper Bollinger Band
    if n >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        distance_from_upper_band = (closing_prices[-1] - upper_band) / upper_band if upper_band != 0 else 0
    else:
        distance_from_upper_band = 0
    features.append(distance_from_upper_band)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Determine historical thresholds based on features
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    threshold_risk_high = 0.7 * historical_std
    threshold_risk_medium = 0.4 * historical_std
    threshold_trend_high = 0.3 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > threshold_risk_high:
        if features[2] > 0:  # If ROC is positive (momentum indicates buy)
            reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        else:
            reward += np.random.uniform(5, 10)  # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)

    if risk_level > threshold_risk_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > threshold_trend_high and risk_level <= threshold_risk_medium:
        momentum_score = features[2]  # Assuming features[2] is ROC
        if trend_direction > threshold_trend_high and momentum_score > 0:  # Bullish condition
            reward += 20  # Positive reward for correct bullish position
        elif trend_direction < -threshold_trend_high and momentum_score < 0:  # Bearish condition
            reward += 20  # Positive reward for correct bearish position

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < threshold_trend_high and risk_level < threshold_risk_medium:
        if features[3] < 0:  # Assuming distance from upper band < 0 indicates oversold/mean reversion opportunity
            reward += 15  # Reward for buying in oversold condition
        elif features[3] > 0:  # Assuming distance from upper band > 0 indicates overbought/mean reversion opportunity
            reward += 15  # Reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < threshold_risk_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds