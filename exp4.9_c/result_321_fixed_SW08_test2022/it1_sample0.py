import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    volumes = s[4:120:6]          # Trading volumes for 20 days

    # Feature 1: Price Change Percentage (last day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Over Last 20 Days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Standard Deviation of Closing Prices (Volatility)
    volatility = np.std(closing_prices) if len(closing_prices) > 0 else 0.0

    # Feature 4: Closing Price vs. 20-Day Moving Average
    moving_average_20 = np.mean(closing_prices) if len(closing_prices) >= 1 else closing_prices[-1]
    closing_vs_moving_avg = closing_prices[-1] - moving_average_20

    # Feature 5: Price Momentum (current closing price - previous closing price)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0

    # Combine the features into a single array
    features = [price_change_pct, average_volume, volatility, closing_vs_moving_avg, price_momentum]
    
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
        reward -= 40.0  # Strong negative for buying in high-risk conditions
        reward += 5.0 * (-features[0])  # Mild positive if price change is negative (sell signal)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for buying in elevated risk conditions

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 10.0  # Reward for positive price change
        else:  # Downtrend
            reward += -features[0] * 10.0  # Reward for negative price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.02:  # Oversold condition
            reward += 10.0  # Encourage buying
        elif features[0] > 0.02:  # Overbought condition
            reward -= 5.0  # Encourage selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))