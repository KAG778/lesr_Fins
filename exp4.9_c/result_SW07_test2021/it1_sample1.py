import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    high_prices = s[2::6]      # Extract high prices
    low_prices = s[3::6]       # Extract low prices

    # Feature 1: Rate of Change over the last 5 days
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Feature 2: Historical Volatility (last 20 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns) if len(returns) >= 5 else 0  # Standard deviation of returns

    # Feature 3: Price Range over the last 20 days
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 else 0

    # Feature 4: Volume Spike (percentage change from the average volume)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 1  # Avoid division by zero
    volume_spike = (volumes[-1] - average_volume) / average_volume if average_volume != 0 else 0

    # Combine features into a single array
    features = [roc, historical_volatility, price_range, volume_spike]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical volatility
    risk_threshold = 0.5  # This could be dynamically calculated as a function of historical data
    trend_threshold = 0.3  # Similarly, this could be based on historical data

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY signals
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        elif trend_direction < -trend_threshold:
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)