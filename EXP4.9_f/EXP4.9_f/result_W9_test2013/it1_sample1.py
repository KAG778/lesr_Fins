import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    features = []

    # Feature 1: Price Change (Percentage Change from Previous Closing)
    price_change = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices * 100
    features.append(price_change[-1])  # Latest price change

    # Feature 2: 5-day Moving Average of Closing Prices
    moving_average = np.convolve(closing_prices, np.ones(5) / 5, mode='valid')
    features.append(moving_average[-1] if len(moving_average) > 0 else 0)  # Latest SMA

    # Feature 3: Volume Change (Percentage Change from Previous Volume)
    volume_change = np.diff(volumes, prepend=volumes[0]) / volumes * 100
    features.append(volume_change[-1])  # Latest volume change

    # Feature 4: Historical Volatility (Standard Deviation of Daily Returns)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    features.append(historical_volatility)

    # Feature 5: Correlation of Returns with a Market Index (e.g., S&P 500)
    # This would require additional market data, but for the sake of this function, 
    # we will use a placeholder. In practice, you would pull this from external data.
    market_returns = np.random.normal(0, 0.01, len(daily_returns))  # Placeholder for market returns
    correlation = np.corrcoef(daily_returns, market_returns)[0, 1] if len(daily_returns) > 1 else 0
    features.append(correlation)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical standard deviations
    std_price_change = np.std(features[:-2])  # Use price changes only for threshold
    std_volume_change = np.std(features[:-1])  # Use volume changes only for threshold

    # Priority 1: Risk Management
    if risk_level > 0.7:
        if features[0] > 0:  # If latest price change is positive
            reward -= 50  # Strong negative for BUY in high risk 
        else:
            reward += 10  # Mild positive for SELL in high risk
    elif risk_level > 0.4:
        if features[0] > 0:
            reward -= 20  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += max(10 * features[0] / std_price_change, 10)  # Align with momentum
        else:
            reward += max(10 * -features[0] / std_price_change, 10)

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Oversold condition
            reward += 15  # Positive reward for buying
        elif features[0] > 0.1:  # Overbought condition
            reward -= 15  # Negative reward for buying

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range