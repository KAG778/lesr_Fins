import numpy as np

def revise_state(s):
    # Extract the necessary pricing and volume data
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    features = []
    
    # Feature 1: Exponential Moving Average (EMA) for the last 10 days
    if len(closing_prices) >= 10:
        ema = np.mean(closing_prices[-10:])  # Simple EMA for simplicity, can be replaced with actual EMA calculation
    else:
        ema = 0.0
    features.append(ema)

    # Feature 2: Rate of Change (ROC) over last 10 days
    roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) > 11 else 0
    features.append(roc)

    # Feature 3: Bollinger Bands (20-day standard deviation)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        bb_width = (upper_band - lower_band) / rolling_mean if rolling_mean != 0 else 0
    else:
        bb_width = 0
    features.append(bb_width)

    # Feature 4: Volume Change Percentage (current vs. previous average)
    if len(volumes) >= 2:
        avg_volume = np.mean(volumes[:-1])
        volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    else:
        volume_change = 0
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_prices = enhanced_s[0:120:6]
    historical_std = np.std(historical_prices)
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20  # Positive reward for upward features
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        # Assuming a mean-reversion scenario
        if enhanced_s[123] < 30:  # Oversold condition
            reward += 15  # Reward for buying
        elif enhanced_s[123] > 70:  # Overbought condition
            reward -= 10  # Penalize for selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]