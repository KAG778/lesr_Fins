import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices (every 6th element)

    # Feature 1: Price Momentum (Rate of Change)
    if len(closing_prices) > 1:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Percentage change
    else:
        momentum = 0
    features.append(momentum)

    # Feature 2: Average Volume over the last 20 days
    volumes = s[4::6]  # Extract volumes
    if len(volumes) >= 20:
        avg_volume = np.mean(volumes[-20:])
    else:
        avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    features.append(avg_volume)

    # Feature 3: Bollinger Bands (Upper and Lower Bands)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
    else:
        upper_band, lower_band = closing_prices[-1], closing_prices[-1]  # Use latest closing price if not enough data
    features.append(upper_band)
    features.append(lower_band)

    # Feature 4: Average True Range (ATR) for volatility
    if len(closing_prices) > 2:
        highs = s[2::6]  # Extract highs
        lows = s[1::6]  # Extract lows
        tr = np.maximum(highs[-1] - lows[-1], highs[-1] - closing_prices[-2], closing_prices[-2] - lows[-1])
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 0
    else:
        atr = 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    mean_risk = 0.5  # Historical mean risk level (example, can be dynamically calculated)
    std_risk = 0.2   # Standard deviation of risk level (example, can be dynamically calculated)
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > (mean_risk + std_risk):
        # Strong negative reward for BUY-aligned features
        reward += -40  # Strong negative reward for high risk
        if features[0] < 0:  # Assuming positive momentum indicates BUY
            reward += 5  # Mild positive reward for selling
        return np.clip(reward, -100, 100)

    elif risk_level > mean_risk:  # Moderate risk
        if features[0] > 0:  # Positive momentum
            reward += -20  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < mean_risk:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += 20  # Positive reward for following the trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < (mean_risk - std_risk):
        if features[0] < 0:  # Oversold condition
            reward += 15  # Reward for buying
        elif features[0] > 0:  # Overbought condition
            reward += -15  # Penalize for chasing breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > (mean_risk + std_risk) and risk_level < mean_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)