import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]          # Trading volumes
    high_prices = s[2:120:6]      # High prices
    low_prices = s[3:120:6]       # Low prices
    
    # Feature 1: Price Momentum (Change in closing price over the last 5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: Average True Range (ATR) for Volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    average_true_range = np.mean(true_ranges) if len(true_ranges) > 0 else 0

    # Feature 4: Volume Spike (compared to the historical average)
    historical_average_volume = np.mean(volumes)
    volume_spike = (volumes[-1] - historical_average_volume) / historical_average_volume if historical_average_volume != 0 else 0

    # Return the features as a numpy array
    return np.array([price_momentum, rsi, average_true_range, volume_spike])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    # Calculate thresholds based on historical standard deviations
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for BUY signals in high risk
        reward += 10  # Mild positive reward for SELL signals in high risk
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals in medium risk

    # Trend following reward
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Bullish
            reward += 30  # Strong positive reward for bullish features
        elif trend_direction < -0.3:  # Bearish
            reward += 30  # Strong positive reward for bearish features

    # Sideways market and mean reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 20  # Reward mean-reversion features (oversold → buy, overbought → sell)

    # High volatility
    if volatility_level > np.std([0.5, 0.6, 0.7]) and risk_level < 0.4:  # Dynamic threshold for volatility
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)