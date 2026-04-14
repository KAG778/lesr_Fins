import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price Momentum (current closing price vs. closing price 5 days ago)
    price_momentum = (closing_prices[0] - closing_prices[5]) / (closing_prices[5] if closing_prices[5] != 0 else 1)
    features.append(price_momentum)

    # Feature 2: Volatility (standard deviation of closing prices over the last 20 days)
    volatility = np.std(closing_prices)
    features.append(volatility)

    # Feature 3: Volume Change (percentage change from average volume)
    average_volume = np.mean(volumes)
    volume_change = (volumes[0] - average_volume) / (average_volume if average_volume != 0 else 1)
    features.append(volume_change)

    # Feature 4: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    macd = short_ema - long_ema
    features.append(macd)

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
        reward -= 40.0  # Strong negative for BUY in high risk
    elif risk_level > 0.4:
        reward -= 10.0  # Mild negative for BUY in moderate risk
    else:
        reward += 10.0  # Mild positive for SELL in low risk

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 20.0  # Use price momentum for reward

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Considered oversold
            reward += 10.0  # Strong positive for mean-reversion BUY
        elif features[0] > 0.1:  # Considered overbought
            reward -= 10.0  # Strong negative for mean-reversion SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))