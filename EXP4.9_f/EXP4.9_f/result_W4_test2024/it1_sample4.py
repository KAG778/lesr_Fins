import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Extracting the necessary OHLCV data
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    opening_prices = s[1:120:6]  # Opening prices for 20 days
    high_prices = s[2:120:6]     # High prices for 20 days
    low_prices = s[3:120:6]      # Low prices for 20 days
    volumes = s[4:120:6]         # Trading volumes for 20 days

    days = len(closing_prices)

    # Feature 1: Price Change Percentage (last day closing to previous closing)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Daily Volume Change (percentage change in volume)
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 and np.all(volumes[:-1] != 0) else 0

    # Feature 3: Last Closing Price Relative to Bollinger Bands
    moving_avg = np.mean(closing_prices)
    std_dev = np.std(closing_prices)
    upper_bb = moving_avg + (2 * std_dev)
    lower_bb = moving_avg - (2 * std_dev)
    price_relative_to_bb = (closing_prices[-1] - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0

    # Feature 4: Volatility (Standard Deviation of Closing Prices)
    volatility = np.std(closing_prices)

    # Feature 5: Price Momentum (change from 10 days ago to now)
    if days > 10:
        price_momentum = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11]
    else:
        price_momentum = 0

    features = [price_change_pct, avg_volume_change, price_relative_to_bb, volatility, price_momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds for relative risk
    historical_std = np.std(enhanced_s[123:])  # Assume features are stored here
    threshold_risk_high = 0.7 * historical_std
    threshold_risk_moderate = 0.4 * historical_std
    threshold_trend = 0.3 * historical_std

    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > threshold_risk_high:
        reward -= 50  # STRONG NEGATIVE reward for BUY-aligned features in high risk
        return reward  # Exit early to prevent further positive rewards
    elif risk_level > threshold_risk_moderate:
        reward -= 20  # Moderate negative reward for BUY signals in moderate risk

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > threshold_trend and risk_level < threshold_risk_moderate:
        if trend_direction > 0:
            reward += 30  # Positive reward for upward trend
        else:
            reward += 30  # Positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < threshold_trend and risk_level < 0.3:
        reward += 20  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < threshold_risk_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]