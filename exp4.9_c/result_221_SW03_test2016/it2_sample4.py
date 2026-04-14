import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: Rate of Change (ROC) - measures the percentage change over the last 5 days
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) >= 6 else 0.0

    # Feature 2: Average True Range (ATR) - measures market volatility
    if len(closing_prices) >= 14:
        high_prices = s[2::6]  # High prices
        low_prices = s[3::6]   # Low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:],
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]),
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # 14-day ATR
    else:
        atr = 0.0  # Not enough data

    # Feature 3: Bollinger Bands - measure volatility and potential price reversals
    if len(closing_prices) >= 20:
        moving_avg = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_avg + (2 * std_dev)
        lower_band = moving_avg - (2 * std_dev)
        bb_width = upper_band - lower_band  # Width of Bollinger Bands
    else:
        bb_width = 0.0  # Not enough data

    # Feature 4: Z-Score of Closing Prices - measures how far the current price is from the mean
    if len(closing_prices) >= 20:
        mean_price = np.mean(closing_prices[-20:])
        std_price = np.std(closing_prices[-20:])
        z_score = (closing_prices[-1] - mean_price) / std_price if std_price != 0 else 0.0
    else:
        z_score = 0.0  # Not enough data

    # Return only new features
    return np.array([roc, atr, bb_width, z_score])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(enhanced_s[123:])  # Use the std of features for thresholds
    risk_thresholds = {
        'low': 0.4 * historical_std,
        'medium': 0.7 * historical_std,
        'high': 1.0 * historical_std
    }

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_thresholds['high']:
        reward -= 50  # Strong negative reward for BUY
        reward += 10   # Mild positive reward for SELL
    elif risk_level > risk_thresholds['medium']:
        reward -= 20  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < risk_thresholds['medium']:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_thresholds['low']:
        z_score = enhanced_s[123 + 3]  # Assuming Z-Score is the fourth feature
        if z_score < -1:  # Oversold condition
            reward += 15  # Reward potential buy
        elif z_score > 1:  # Overbought condition
            reward += 15  # Reward potential sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std and risk_level < risk_thresholds['medium']:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)