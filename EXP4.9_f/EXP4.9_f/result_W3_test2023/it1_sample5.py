import numpy as np

def revise_state(s):
    # Extract closing prices, high prices, and low prices
    closing_prices = s[0::6][:20]  # Closing prices
    high_prices = s[2::6][:20]      # High prices
    low_prices = s[3::6][:20]       # Low prices
    volumes = s[4::6][:20]          # Trading volumes

    # Feature 1: Rate of Change (momentum indicator)
    if len(closing_prices) >= 6:
        roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        roc = 0.0

    # Feature 2: Average True Range (ATR) for volatility
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.0  # 14-day ATR

    # Feature 3: Mean Reversion Indicator (Z-score of closing prices)
    moving_average = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else 0.0
    std_dev = np.std(closing_prices[-14:]) if len(closing_prices) >= 14 else 1e-6  # Avoid division by zero
    z_score = (closing_prices[-1] - moving_average) / std_dev

    # Feature 4: Volume Change (current volume vs average volume over the last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1e-6
    volume_change = volumes[-1] / avg_volume - 1  # Percentage change

    features = [roc, atr, z_score, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Calculate dynamic thresholds based on standard deviations of features
    thresholds = {
        "momentum": np.std(features[0:1]) * 2,  # Using momentum feature
        "atr": np.std(features[1:2]) * 1.5,     # Using ATR feature
        "z_score": 1.0,                         # Fixed for Z-score
        "volume_change": np.std(features[3:4]) * 1.5  # Using volume change
    }

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > thresholds["momentum"]:  # If momentum is bullish
            reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        else:
            reward += np.random.uniform(5, 10)  # Mild positive for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > thresholds["momentum"]:  # Bullish trend
            reward += 20  # Positive reward for correct bullish bet
        elif trend_direction < 0 and features[0] < -thresholds["momentum"]:  # Bearish trend
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -thresholds["z_score"]:  # Oversold condition
            reward += 15  # Reward buying
        elif features[2] > thresholds["z_score"]:  # Overbought condition
            reward += -15  # Penalize for breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds