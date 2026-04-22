import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (20 days)
    volumes = s[4:120:6]          # Trading volumes for 20 days
    days = len(closing_prices)

    # Feature 1: Price Change Percentage (last day closing to previous closing)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Daily Volume Change (percentage change in volume)
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 and np.all(volumes[:-1] != 0) else 0

    # Feature 3: Bollinger Band Width (to measure volatility)
    moving_avg = np.mean(closing_prices)
    std_dev = np.std(closing_prices)
    bb_width = (std_dev / moving_avg) if moving_avg != 0 else 0

    # Feature 4: Price Momentum (last closing price vs. N-day average)
    n_days = 20
    if days >= n_days:
        n_day_avg = np.mean(closing_prices[-n_days:])
        price_momentum = (closing_prices[-1] - n_day_avg) / n_day_avg if n_day_avg != 0 else 0
    else:
        price_momentum = 0

    # Feature 5: Exponential Moving Average (EMA) of closing prices
    short_ema = np.mean(closing_prices[-5:]) if days >= 5 else 0  # 5-day EMA
    long_ema = np.mean(closing_prices[-20:]) if days >= 20 else 0  # 20-day EMA
    ema_difference = short_ema - long_ema

    features = [price_change_pct, avg_volume_change, bb_width, price_momentum, ema_difference]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds for risk levels
    risk_threshold_high = 0.7
    risk_threshold_mid = 0.4
    volatility_threshold = 0.6
    
    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # STRONG NEGATIVE for BUY-aligned features
        return np.clip(reward, -100, 100)
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_mid:
        if trend_direction > 0:
            reward += 25  # Strong positive reward for upward trend
        else:
            reward += 25  # Strong positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]