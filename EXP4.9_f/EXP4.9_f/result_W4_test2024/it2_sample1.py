import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices for 20 days
    volumes = s[4:120:6]          # Trading volumes for 20 days
    days = len(closing_prices)

    # Feature 1: Rate of Change (momentum)
    if days >= 5:
        momentum = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5]
    else:
        momentum = 0

    # Feature 2: Average True Range (ATR) - measure of volatility
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    
    if days >= 14:
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.abs(high_prices[1:] - closing_prices[:-1]), 
                        np.abs(low_prices[1:] - closing_prices[:-1]))
        atr = np.mean(tr[-14:])  # 14-day ATR
    else:
        atr = 0

    # Feature 3: Price Relative to 50-day Moving Average
    if days >= 50:
        moving_avg_50 = np.mean(closing_prices[-50:])
        price_relative_to_ma50 = (closing_prices[-1] - moving_avg_50) / moving_avg_50
    else:
        price_relative_to_ma50 = 0

    # Feature 4: Bollinger Band Width (to measure volatility)
    moving_avg = np.mean(closing_prices)
    std_dev = np.std(closing_prices)
    bb_width = (std_dev / moving_avg) if moving_avg != 0 else 0

    features = [momentum, atr, price_relative_to_ma50, bb_width]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Use features to calculate a standard deviation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # STRONG NEGATIVE for BUY-aligned features
        # Potential mild positive for SELL-aligned features (to manage risk)
        reward += 10 if trend_direction < 0 else 0
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Positive reward for trend-following

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]