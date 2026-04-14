import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices

    # Feature 1: Price Change Percentage (latest to previous)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Over Last 20 Days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Historical Volatility (standard deviation of closing prices)
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 0 else 0.0

    # Feature 4: Price Momentum (current closing price - 5-day moving average)
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    price_momentum = closing_prices[-1] - moving_average_5

    # Feature 5: Relative Strength Index (RSI) for Mean Reversion
    price_changes = np.diff(closing_prices)  # Calculate price changes
    gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
    loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [price_change_pct, average_volume, historical_volatility, price_momentum, rsi]
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
        # Mild positive for SELL if price change is negative
        if features[0] < 0:
            reward += 10.0 * abs(features[0])  # Encouraging selling when price change is negative
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY in elevated risk

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Reward for positive price change
        else:  # Downtrend
            reward += -features[0] * 20.0  # Reward for negative price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[4] < 30:  # Oversold condition (RSI)
            reward += 15.0  # Encourage buying
        elif features[4] > 70:  # Overbought condition (RSI)
            reward += -10.0  # Encourage selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))