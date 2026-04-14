import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes

    # Feature 1: Price Change Percentage (latest to previous)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Over Last 20 Days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Price Change Standard Deviation (last 20 days)
    price_changes = [(closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1] for i in range(1, len(closing_prices))]
    price_change_std = np.std(price_changes) if len(price_changes) > 0 else 0.0

    # Feature 4: Price Momentum (current closing price - 5-day moving average)
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    price_momentum = closing_prices[-1] - moving_average_5

    # Feature 5: Volatility Level (standard deviation of closing prices)
    volatility = np.std(closing_prices)

    # Return the features as a numpy array
    return np.array([price_change_pct, average_volume, price_change_std, price_momentum, volatility])

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
        # Mild positive for SELL if price change is negative
        if features[0] < 0:
            reward += 5.0 * abs(features[0])  # Encouraging selling when price change is negative
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY in elevated risk

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 50.0  # Price change positively aligned
        else:  # Downtrend
            reward += -features[0] * 50.0  # Price change negatively aligned

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.02:  # Oversold
            reward += 10.0  # Encourage buying
        elif features[0] > 0.02:  # Overbought
            reward += -5.0  # Penalize for chasing breakouts

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))