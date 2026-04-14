import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]
    volumes = s[4:120:6]

    # Feature 1: 5-Day Moving Average
    ma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    
    # Feature 2: Bollinger Bands Width (20-day)
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_avg + (2 * std_dev)
        lower_band = moving_avg - (2 * std_dev)
        bollinger_width = upper_band - lower_band
    else:
        bollinger_width = np.nan
    
    # Feature 3: Rate of Change (ROC) over the last 5 days
    if len(closing_prices) >= 6:
        roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        roc = np.nan
    
    # Feature 4: Volume Change Percentage (last day vs. average of last 5 days)
    if len(volumes) >= 6:
        avg_volume = np.mean(volumes[-6:-1])
        volume_change_pct = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    else:
        volume_change_pct = 0.0  # Default to 0 if not enough data

    # Feature 5: Exponential Moving Average (EMA) - 12 Days
    ema_12 = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else np.nan
    
    # Return features as a numpy array
    return np.array([ma_5, bollinger_width, roc, volume_change_pct, ema_12])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for dynamic thresholds
    features = enhanced_s[123:]  # Get the features from the enhanced state
    mean_risk = np.mean(features)  # Mean of the features as a proxy for historical risk
    std_risk = np.std(features)     # Standard deviation of features
    risk_threshold_high = mean_risk + std_risk
    risk_threshold_low = mean_risk - std_risk
    trend_threshold = 0.3  # Relative threshold for trend direction

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY
        reward += 10   # Mild positive reward for SELL
    elif risk_level > risk_threshold_low:
        reward -= 20  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold:
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        # Assuming we have features that indicate mean reversion (like Z-Score from previous features)
        z_score = (enhanced_s[123] - np.mean(features)) / np.std(features) if np.std(features) != 0 else 0
        if z_score < -1:  # Oversold condition
            reward += 10  # Reward potential buy
        elif z_score > 1:  # Overbought condition
            reward += 10  # Reward potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > mean_risk + std_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return float(np.clip(reward, -100, 100))