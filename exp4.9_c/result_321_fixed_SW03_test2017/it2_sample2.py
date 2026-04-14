import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices, volumes, etc.
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes
    
    # Feature 1: Price Change Rate (current vs. average of last 5 days)
    price_change_rate = (closing_prices[0] - np.mean(closing_prices[1:6])) / (np.mean(closing_prices[1:6]) if np.mean(closing_prices[1:6]) != 0 else 1)
    features.append(price_change_rate)
    
    # Feature 2: Historical Volatility (Standard deviation of closing prices over the last 20 days)
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.0
    features.append(historical_volatility)
    
    # Feature 3: Relative Volume (current volume vs. average volume over the last 20 days)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 1.0
    current_volume = volumes[-1] if len(volumes) > 0 else 0.0
    relative_volume = (current_volume - average_volume) / (average_volume if average_volume != 0 else 1)
    features.append(relative_volume)
    
    # Feature 4: Price Momentum (current closing price vs. moving average of last 5 days)
    moving_average = np.mean(closing_prices[:5]) if len(closing_prices) >= 5 else 0
    price_momentum = (closing_prices[0] - moving_average) / (moving_average if moving_average != 0 else 1)
    features.append(price_momentum)
    
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
        if features[0] < 0:  # If price change rate is negative, consider a mild positive for SELL
            reward += 10.0  # Mild positive for SELL in high risk
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[3] * 15.0  # Use price momentum for reward

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