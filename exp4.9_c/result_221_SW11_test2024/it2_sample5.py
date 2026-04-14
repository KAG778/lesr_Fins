import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes
    n = len(closing_prices)

    features = []

    # Feature 1: Average True Range (ATR) over the last 14 days (volatility measure)
    if n >= 14:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # Average of the True Range over the last 14 days
        features.append(atr)
    else:
        features.append(0.0)

    # Feature 2: Rate of Change (ROC) over the last 5 days
    if n > 5:
        roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
        features.append(roc)
    else:
        features.append(0.0)

    # Feature 3: Distance from the last 20-day Moving Average (MA)
    if n >= 20:
        moving_average = np.mean(closing_prices[-20:])
        distance_from_ma = (closing_prices[-1] - moving_average) / moving_average  # Normalize
        features.append(distance_from_ma)
    else:
        features.append(0.0)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical thresholds for relative measures
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold_high = 0.3 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if features[1] > 0:  # If ROC is positive, indicates BUY
            reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        else:
            reward += np.random.uniform(5, 10)  # Mild positive reward for SELL-aligned features
        return np.clip(reward, -100, 100)

    if risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > trend_threshold_high and risk_level < risk_threshold_medium:
        momentum_score = features[1]  # Assuming ROC indicates momentum
        if trend_direction > trend_threshold_high and momentum_score > 0:  # Bullish condition
            reward += 10  # Positive reward for correct bullish position
        elif trend_direction < -trend_threshold_high and momentum_score < 0:  # Bearish condition
            reward += 10  # Positive reward for correct bearish position

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold_high and risk_level < 0.3:
        if features[2] < 0:  # If price is below moving average (oversold condition)
            reward += 15  # Reward for buying in oversold condition
        elif features[2] > 0:  # If price is above moving average (overbought condition)
            reward += 15  # Reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is capped within the specified range