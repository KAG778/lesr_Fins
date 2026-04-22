import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices for 20 days
    opening_prices = s[1:120:6]  # Opening prices for 20 days
    high_prices = s[2:120:6]     # High prices for 20 days
    low_prices = s[3:120:6]      # Low prices for 20 days
    volumes = s[4:120:6]         # Trading volumes for 20 days

    # Feature 1: Average True Range (ATR) for measuring volatility
    high_low = high_prices - low_prices
    high_close = np.abs(high_prices[1:] - closing_prices[:-1])
    low_close = np.abs(low_prices[1:] - closing_prices[:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 2: Price Momentum over 5 days relative to historical volatility
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / atr if atr != 0 else 0

    # Feature 3: Z-score of the last closing price relative to its moving average
    moving_avg = np.mean(closing_prices)
    std_dev = np.std(closing_prices)
    z_score = (closing_prices[-1] - moving_avg) / std_dev if std_dev != 0 else 0

    # Feature 4: Volume Change Rate (last day vs previous day)
    volume_change_rate = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0

    features = [atr, price_momentum, z_score, volume_change_rate]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0
    
    # Calculate relative thresholds based on historical data
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY-aligned features during high risk
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 25 if trend_direction > 0 else 25  # Positive reward for trend-aligned momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]