import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Recent Price Momentum (Current closing price - 5-day moving average)
    if len(closing_prices) > 5:
        recent_price_momentum = closing_prices[-1] - np.mean(closing_prices[-5:])
    else:
        recent_price_momentum = 0.0
    
    features.append(recent_price_momentum)

    # Feature 2: Historical Volatility (Standard deviation of closing prices over the last 20 days)
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.0
    features.append(historical_volatility)

    # Feature 3: Volume Spike (Current volume vs. average volume over the last 20 days)
    average_volume = np.mean(volumes) if len(volumes) > 0 else 1.0
    current_volume = volumes[-1] if len(volumes) > 0 else 0.0
    volume_spike = (current_volume - average_volume) / average_volume
    features.append(volume_spike)

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
        # Strong negative penalty for BUY signals in high risk
        reward -= 40.0  
    elif risk_level > 0.4:
        # Mild negative penalty for BUY signals in moderate risk
        reward -= 10.0  

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 15.0  # Align reward with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward based on mean-reversion logic
        if features[0] < -0.05:  # Oversold condition
            reward += 10.0  # Strong positive for mean-reversion buy signal
        elif features[0] > 0.05:  # Overbought condition
            reward -= 10.0  # Strong negative for mean-reversion sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))