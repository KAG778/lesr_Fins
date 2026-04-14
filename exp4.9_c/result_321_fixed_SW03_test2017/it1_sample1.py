import numpy as np

def revise_state(s):
    features = []

    # Extract relevant price and volume information
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Price Momentum (current closing price vs. closing price 5 days ago)
    if len(closing_prices) > 5:
        price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        price_momentum = 0.0

    # Feature 2: Volatility (standard deviation of closing prices over the last 20 days)
    volatility = np.std(closing_prices)

    # Feature 3: Volume Change (percentage change from average volume)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0
    volume_change = (volumes[-1] - average_volume) / (average_volume if average_volume != 0 else 1)

    # Feature 4: Price Range (max high - min low over the last 20 days)
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 and len(low_prices) > 0 else 0

    # Collect features
    features = [price_momentum, volatility, volume_change, price_range]
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
        reward -= 50.0  # Strong negative for BUY in high risk
        if features[0] < 0:  # If price momentum is negative
            reward += 20.0  # Mild positive for SELL in high risk
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY in moderate risk

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 25.0  # Momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition
            reward += 10.0  # Positive for mean-reversion BUY signal
        elif features[0] > 0.1:  # Overbought condition
            reward -= 10.0  # Negative for mean-reversion SELL signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))