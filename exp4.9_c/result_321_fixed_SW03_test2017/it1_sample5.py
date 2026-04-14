import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices, volumes, and calculate necessary features
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price Change (current closing price vs. closing price 5 days ago)
    price_change = (closing_prices[0] - closing_prices[5]) / (closing_prices[5] if closing_prices[5] != 0 else 1)

    # Feature 2: Standardized Volatility (std dev of closing prices over the last 20 days)
    volatility = np.std(closing_prices)

    # Feature 3: Volume Change (percentage change from 20-day average volume)
    average_volume = np.mean(volumes)
    volume_change = (volumes[0] - average_volume) / (average_volume if average_volume != 0 else 1)

    # Feature 4: Average True Range (ATR) to capture price movement volatility
    true_ranges = np.abs(np.maximum(closing_prices[1:] - closing_prices[:-1], 0))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Feature 5: Price Momentum (current closing price vs. moving average of last 5 days)
    moving_average = np.mean(closing_prices[:5]) if len(closing_prices) >= 5 else 0
    price_momentum = (closing_prices[0] - moving_average) / (moving_average if moving_average != 0 else 1)

    features = [price_change, volatility, volume_change, atr, price_momentum]
    
    return np.array(features)

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
        if features[0] < 0:  # If price momentum is negative in a dangerous situation
            reward += 10.0  # Mild positive for SELL
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[4] * 10.0  # Use price momentum for reward

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition (price change)
            reward += 5.0  # Mild positive for mean-reversion BUY
        elif features[0] > 0.1:  # Overbought condition (price change)
            reward -= 5.0  # Mild negative for mean-reversion SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))