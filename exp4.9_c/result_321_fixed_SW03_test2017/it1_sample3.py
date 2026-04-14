import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state (OHLCV for 20 days)
    
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]  # Extract trading volumes
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]  # Extract low prices

    # Feature 1: Price Momentum (current closing price vs. closing price 5 days ago)
    price_momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) >= 6 else 0

    # Feature 2: Relative Volatility (standard deviation of closing prices over the last 20 days)
    volatility = np.std(closing_prices) if len(closing_prices) > 0 else 0

    # Feature 3: Volume Change (percentage change from average volume)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 1  # Avoid division by zero
    current_volume_change = (volumes[0] - average_volume) / average_volume

    # Feature 4: Price Range (High - Low over the last 20 days)
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 and len(low_prices) > 0 else 0

    # Return the computed features as a numpy array
    return np.array([price_momentum, volatility, current_volume_change, price_range])


def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 1: Mild positive for SELL when in high-risk situation
    if risk_level > 0.7:
        reward += 20.0  # Strong positive for SELL

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 15.0  # Reward based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Negative momentum could indicate oversold
            reward += 5.0  # Buy signal
        elif features[0] > 0:  # Positive momentum could indicate overbought
            reward -= 5.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))