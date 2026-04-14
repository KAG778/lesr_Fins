import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices (day i at index 6*i)
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    volumes = s[4:120:6]          # Extract volumes

    # Feature 1: Exponential Moving Average (EMA) for the last 10 periods
    ema_period = 10
    ema = np.mean(closing_prices[-ema_period:]) if len(closing_prices) >= ema_period else closing_prices[-1]
    
    # Feature 2: Bollinger Bands
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
    else:
        upper_band, lower_band = closing_prices[-1] * 1.05, closing_prices[-1] * 0.95  # Fallback values

    # Feature 3: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0

    # Return the features as a 1D numpy array
    features = [ema, upper_band, lower_band, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical std deviation for relative thresholds
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        if features[0] > 0:  # Assuming ema indicates a positive outlook
            reward = np.random.uniform(-100, -50)  # Strong negative reward
        else:
            reward = np.random.uniform(10, 20)  # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward = np.random.uniform(-30, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > features[1]:  # EMA above upper band indicates bullish
            reward += 20  # Positive reward for trend alignment
        elif trend_direction < -0.3 and features[0] < features[2]:  # EMA below lower band indicates bearish
            reward += 20  # Positive reward for trend alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < features[2]:  # EMA below lower band (oversold)
            reward += 10  # Reward for buying in oversold conditions
        elif features[0] > features[1]:  # EMA above upper band (overbought)
            reward += 10  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward