import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices for the last 20 days
    opening_prices = s[1::6]   # Opening prices for the last 20 days
    high_prices = s[2::6]      # High prices for the last 20 days
    low_prices = s[3::6]       # Low prices for the last 20 days
    volumes = s[4::6]          # Trading volumes for the last 20 days

    # Feature 1: Percentage Price Change from Opening to Last Close
    price_change = (closing_prices[-1] - opening_prices[-1]) / opening_prices[-1] if opening_prices[-1] != 0 else 0.0

    # Feature 2: Average Volume Change (Percentage change in volume)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    volume_change = (volumes[-1] - average_volume) / average_volume if average_volume != 0 else 0.0

    # Feature 3: Price Range (Normalized by previous close)
    price_range = (high_prices[-1] - low_prices[-1]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 4: 14-day Historical Volatility (Standard Deviation of Daily Returns)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] if len(closing_prices) > 1 else np.array([0.0])
    historical_volatility = np.std(daily_returns)

    features = [
        price_change,        # Feature 1
        volume_change,       # Feature 2
        price_range,         # Feature 3
        historical_volatility # Feature 4
    ]
    
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
        reward -= 40.0  # Strong negative for BUY-aligned features
        # Mild positive for SELL if price change is negative
        if features[0] < 0:
            reward += 5.0
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Positive reward for positive price change
        else:  # Downtrend
            reward += -features[0] * 20.0  # Positive reward for negative price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 10.0  # Reward for potential buy
        elif features[0] > 0:  # Overbought condition
            reward += 5.0  # Reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in uncertain markets

    return float(np.clip(reward, -100, 100))