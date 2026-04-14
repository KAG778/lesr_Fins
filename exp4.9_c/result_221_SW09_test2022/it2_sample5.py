import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Price Momentum (Rate of Change, 1-day)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Historical Volatility (Standard Deviation of Returns over a window)
    returns = np.diff(closing_prices)
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0  # Last 20 returns

    # Feature 3: Change in Volume Relative to Moving Average
    if len(volumes) >= 10:
        volume_moving_average = np.mean(volumes[-10:])
        volume_change = (volumes[-1] - volume_moving_average) / volume_moving_average if volume_moving_average != 0 else 0
    else:
        volume_change = 0

    # Feature 4: Relative Strength Index (RSI) over a 14-day period
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rsi = 100 - (100 / (1 + (average_gain / average_loss))) if average_loss != 0 else 100

    # Feature 5: Volume Volatility (Standard Deviation of volume over a window)
    volume_returns = np.diff(volumes)
    volume_volatility = np.std(volume_returns[-20:]) if len(volume_returns) >= 20 else 0  # Last 20 volume returns

    features = [price_momentum, historical_volatility, volume_change, rsi, volume_volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical data
    risk_threshold_high = 0.7  # Example threshold for high risk
    risk_threshold_moderate = 0.4  # Example threshold for moderate risk
    trend_threshold = 0.3  # Example threshold for trend direction
    volatility_threshold = 0.6  # Example threshold for high volatility

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY signals
        reward += np.random.uniform(5, 15)    # Mild positive for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(5, 15)  # Mild negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        reward += np.random.uniform(10, 20) if trend_direction > 0 else -np.random.uniform(10, 20)  # Reward for momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > volatility_threshold and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)
    
    return reward