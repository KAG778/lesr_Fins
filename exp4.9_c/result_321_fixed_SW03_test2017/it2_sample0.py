import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices, volumes, and calculate necessary features
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price Change Rate (current vs. average of last 5 days)
    price_change_rate = (closing_prices[0] - np.mean(closing_prices[1:6])) / (np.mean(closing_prices[1:6]) if np.mean(closing_prices[1:6]) != 0 else 1)

    # Feature 2: Standardized Volatility (std dev of closing prices over the last 20 days)
    volatility = np.std(closing_prices) if len(closing_prices) > 0 else 0

    # Feature 3: Relative Volume Change (current volume vs. average volume)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 1
    relative_volume = (volumes[0] - average_volume) / (average_volume if average_volume != 0 else 1)

    # Feature 4: Average True Range (ATR) to capture price movement volatility
    true_ranges = np.abs(np.maximum(closing_prices[1:] - closing_prices[:-1], 0))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Feature 5: 10-Day Momentum (momentum over a 10-day window)
    momentum_period = 10
    n_day_momentum = closing_prices[0] - closing_prices[momentum_period] if len(closing_prices) >= momentum_period else 0.0

    features = [price_change_rate, volatility, relative_volume, atr, n_day_momentum]
    
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
        reward -= 40.0  # Strong negative for BUY signals in high risk
        if features[0] < 0:  # If price change rate is negative, consider a mild positive for SELL
            reward += 10.0  # Mild positive for SELL in high risk
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[4] * 10.0  # Use N-Day momentum for reward

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition based on price change rate
            reward += 10.0  # Strong positive for mean-reversion BUY
        elif features[0] > 0.1:  # Overbought condition
            reward -= 10.0  # Strong negative for mean-reversion SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce the reward magnitude by 50%

    return float(np.clip(reward, -100, 100))