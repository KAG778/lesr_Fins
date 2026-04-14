import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes
    n = len(closing_prices)

    features = []

    # Feature 1: Rate of Change (ROC) over the last 14 days
    if n >= 14:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if closing_prices[-15] != 0 else 0  # Avoid division by zero
        features.append(roc)
    else:
        features.append(0)

    # Feature 2: Average True Range (ATR) over the last 14 days (volatility measure)
    if n >= 15:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # Calculate ATR
        features.append(atr)
    else:
        features.append(0)

    # Feature 3: Price Momentum (current close - close 20 days ago)
    if n > 20:
        price_momentum = closing_prices[-1] - closing_prices[-21]
        features.append(price_momentum)
    else:
        features.append(0)

    # Feature 4: Distance from EMA (Exponential Moving Average) over the last 10 days
    if n >= 10:
        ema = np.mean(closing_prices[-10:])  # Simple EMA approximation
        distance_from_ema = (closing_prices[-1] - ema) / ema if ema != 0 else 0
        features.append(distance_from_ema)
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

    # Determine thresholds for risk based on historical standard deviation of features
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if features[0] > 0:  # Assuming positive ROC indicates BUY
            reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        else:
            reward += np.random.uniform(5, 10)    # Mild positive reward for SELL
        return np.clip(reward, -100, 100)

    if risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals
        return np.clip(reward, -100, 100)

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        momentum_score = features[0]  # Assuming features[0] is ROC
        if trend_direction > trend_threshold and momentum_score > 0:  # Bullish condition
            reward += 20  # Positive reward for correct bullish position
        elif trend_direction < -trend_threshold and momentum_score < 0:  # Bearish condition
            reward += 20  # Positive reward for correct bearish position

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[3] < 0:  # If distance from EMA is negative (price below EMA)
            reward += 10  # Reward for buying in oversold condition
        elif features[3] > 0:  # If distance from EMA is positive (price above EMA)
            reward += 10  # Reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds