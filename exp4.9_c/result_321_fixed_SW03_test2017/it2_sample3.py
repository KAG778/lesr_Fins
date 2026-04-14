import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices

    # Feature 1: Price Change Rate (current vs. average of last 5 days)
    price_change_rate = (closing_prices[0] - np.mean(closing_prices[1:6])) / (np.mean(closing_prices[1:6]) if np.mean(closing_prices[1:6]) != 0 else 1)
    features.append(price_change_rate)

    # Feature 2: Relative Volume (current volume vs. average volume)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 1.0
    relative_volume = (volumes[0] - average_volume) / (average_volume if average_volume != 0 else 1)
    features.append(relative_volume)

    # Feature 3: N-Day Momentum (momentum over a 10-day window)
    momentum_period = 10
    n_day_momentum = (closing_prices[0] - closing_prices[momentum_period]) / (closing_prices[momentum_period] if len(closing_prices) >= momentum_period else 1)
    features.append(n_day_momentum)

    # Feature 4: Price Range (High - Low over the last 20 days)
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 and len(low_prices) > 0 else 0
    features.append(price_range)

    # Feature 5: Historical Volatility (standard deviation of closing prices over the last 20 days)
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.0
    features.append(historical_volatility)

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
        if features[0] < 0:  # If price change rate is negative, consider a mild positive for SELL
            reward += 10.0  # Mild positive for SELL in high risk
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[2] * 15.0  # Use N-Day momentum for reward

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