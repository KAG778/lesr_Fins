import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices (every 6th element starting from index 0)
    volumes = s[4::6]  # Extract trading volumes (every 6th element starting from index 4)

    # Feature 1: Price Momentum (Rate of Change over the last 3 days)
    price_momentum = (closing_prices[-1] - closing_prices[-4]) / closing_prices[-4] if closing_prices[-4] != 0 else 0

    # Feature 2: 10-Day Historical Volatility (Standard Deviation of Returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0

    # Feature 3: Volume Change (Percentage change from the average of last 5 days to the average of the previous 5 days)
    if len(volumes) >= 10:
        avg_volume_current = np.mean(volumes[-5:])
        avg_volume_previous = np.mean(volumes[-10:-5])
        volume_change = (avg_volume_current - avg_volume_previous) / avg_volume_previous if avg_volume_previous != 0 else 0
    else:
        volume_change = 0

    # Feature 4: 14-Day Exponential Moving Average (EMA) of Closing Prices
    ema_span = 14
    if len(closing_prices) >= ema_span:
        k = 2 / (ema_span + 1)
        ema = closing_prices[-ema_span]
        for price in closing_prices[-ema_span:]:
            ema = (price - ema) * k + ema
    else:
        ema = 0

    features = [price_momentum, historical_volatility, volume_change, ema]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Define relative thresholds based on historical data
    risk_thresholds = np.array([0.2, 0.5, 0.7])  # Example thresholds for risk levels
    trend_threshold = 0.3  # Example threshold for trend direction
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_thresholds[2]:  # High risk
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(10, 20)   # Mild positive for SELL signals
    elif risk_level > risk_thresholds[1]:  # Moderate risk
        reward -= np.random.uniform(5, 15)    # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < risk_thresholds[1]:  # Trend is strong and risk is low
        reward += np.random.uniform(10, 20) if trend_direction > 0 else -np.random.uniform(10, 20)  # Positive for momentum alignment

    # Priority 3: Sideways Market / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < risk_thresholds[0]:  # Sideways and low risk
        reward += np.random.uniform(5, 15)  # Reward for mean reversion

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_thresholds[1]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward