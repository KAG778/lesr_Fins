import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    # Feature 1: Price Momentum (1-day rate of change)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: 20-Day Historical Volatility (Standard Deviation of Returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

    # Feature 3: Volume Momentum (1-day rate of change)
    volume_momentum = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0

    # Feature 4: 14-Day Average True Range (ATR) as a volatility measure
    high_prices = s[1::6]  # Extract high prices
    low_prices = s[2::6]   # Extract low prices
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Add feature 5: Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    average_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    average_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rsi = 100 - (100 / (1 + average_gain / average_loss)) if average_loss != 0 else 100

    features = [price_momentum, historical_volatility, volume_momentum, atr, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical standard deviation
    risk_threshold_high = 0.7  # High risk threshold
    risk_threshold_moderate = 0.4  # Moderate risk threshold
    trend_threshold = 0.3  # Trend direction threshold
    volatility_threshold = 0.6  # High volatility threshold

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY signals
        reward += np.random.uniform(10, 20)   # Mild positive for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(5, 15)  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        reward += np.random.uniform(10, 20) if trend_direction > 0 else -np.random.uniform(10, 20)

    # Priority 3: Sideways Market / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean reversion

    # Priority 4: High Volatility
    if volatility_level > volatility_threshold and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward