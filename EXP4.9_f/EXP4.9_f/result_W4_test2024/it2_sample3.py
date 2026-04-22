import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices for 20 days
    high_prices = s[2:120:6]     # High prices for 20 days
    low_prices = s[3:120:6]      # Low prices for 20 days
    volumes = s[4:120:6]         # Trading volumes for 20 days
    
    # Feature 1: Average True Range (ATR) to measure volatility
    def calculate_atr(high, low, close, window=14):
        tr = np.maximum(high - low, np.maximum(np.abs(high - close[:-1]), np.abs(low - close[:-1])))
        return np.mean(tr[-window:]) if len(tr) >= window else 0

    atr_value = calculate_atr(high_prices, low_prices, closing_prices)

    # Feature 2: Price Relative to a 50-day Moving Average
    if len(closing_prices) >= 50:
        moving_avg_50 = np.mean(closing_prices[-50:])
        price_relative_to_ma50 = (closing_prices[-1] - moving_avg_50) / moving_avg_50
    else:
        price_relative_to_ma50 = 0

    # Feature 3: Recent Price Momentum (Rate of Change)
    momentum = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if closing_prices[-5] != 0 else 0

    # Feature 4: Volume Change Rate (percentage change in volume)
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 and np.all(volumes[:-1] != 0) else 0

    # Feature 5: Bollinger Band Width to measure volatility
    moving_avg = np.mean(closing_prices)
    std_dev = np.std(closing_prices)
    bb_width = (std_dev / moving_avg) if moving_avg != 0 else 0

    features = [atr_value, price_relative_to_ma50, momentum, avg_volume_change, bb_width]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Standard deviation of features
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # STRONG NEGATIVE for BUY-aligned features
        if trend_direction < 0:
            reward += 10  # Mild positive for SELL-aligned features
        return float(np.clip(reward, -100, 100))
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 25 * np.sign(trend_direction)  # Reward for momentum alignment
    
    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]