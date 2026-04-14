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

    # Feature 2: Average True Range (ATR)
    if n > 1:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) > 0 else 0
        features.append(atr)
    else:
        features.append(0)

    # Feature 3: Rate of Change (ROC) of closing prices
    if n > 1:
        roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    else:
        roc = 0
    features.append(roc)

    # Feature 4: Current Price Relative to Bollinger Bands
    if n >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        # Normalize the current price relative to the bands
        price_relative_to_bands = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0
        features.append(price_relative_to_bands)
    else:
        features.append(0)

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
    trend_threshold = 0.3 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if features[2] > 0:  # If ROC indicates a bullish feature
            reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        else:
            reward += np.random.uniform(5, 10)  # Mild positive reward for SELL-aligned features
        
    elif risk_level > risk_threshold_medium:
        if features[2] > 0:  # If ROC indicates a bullish feature
            reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level <= risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += features[2] * 20  # Reward based on ROC
        elif trend_direction < 0:  # Downtrend
            reward += -features[2] * 20  # Penalize based on ROC

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= trend_threshold and risk_level < risk_threshold_medium:
        if features[3] < 0:  # If price is below the lower Bollinger Band
            reward += 15  # Reward for buying in oversold condition
        elif features[3] > 0:  # If price is above the upper Bollinger Band
            reward += 15  # Reward for selling in overbought condition

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds