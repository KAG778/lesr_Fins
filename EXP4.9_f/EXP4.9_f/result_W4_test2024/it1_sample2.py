import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    high_prices = s[2:120:6]     # High prices for 20 days
    low_prices = s[3:120:6]      # Low prices for 20 days
    volumes = s[4:120:6]         # Trading volumes for 20 days
    
    # Feature 1: Price Change Percentage (last day closing to previous closing)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Daily Volume Change (percentage change in volume)
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 and np.all(volumes[:-1] != 0) else 0

    # Feature 3: Bollinger Band Relative Position
    moving_avg = np.mean(closing_prices)
    std_dev = np.std(closing_prices)
    upper_bb = moving_avg + (2 * std_dev)
    lower_bb = moving_avg - (2 * std_dev)
    price_relative_to_bb = (closing_prices[-1] - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0

    # Feature 4: Volatility (standard deviation of closing prices)
    volatility = np.std(closing_prices)

    # Feature 5: Momentum (using rate of change)
    momentum = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if closing_prices[-5] != 0 else 0

    features = [price_change_pct, avg_volume_change, price_relative_to_bb, volatility, momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical data
    risk_threshold_high = 0.7  # This could be adapted based on historical metrics
    risk_threshold_medium = 0.4  
    trend_threshold = 0.3
    
    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # STRONG NEGATIVE for BUY-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= 25  # Mild negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > 0:
            reward += 30  # Strong positive reward for upward momentum
        else:
            reward += 30  # Strong positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]