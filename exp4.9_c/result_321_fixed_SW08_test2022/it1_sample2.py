import numpy as np

def revise_state(s):
    # Extracting the closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    volumes = s[4:120:6]  # Trading volumes for 20 days

    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Over Last 20 Days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Volatility as Standard Deviation of Prices
    volatility = np.std(closing_prices)

    # Feature 4: Closing Price vs. 5-Day Moving Average
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    closing_vs_moving_avg = closing_prices[-1] - moving_average_5

    # Feature 5: Relative Strength Index (RSI) for Mean Reversion
    price_changes = np.diff(closing_prices)
    gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
    loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [price_change_pct, average_volume, volatility, closing_vs_moving_avg, rsi]
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
        reward -= 50.0  # Strong negative for BUY-aligned features
        reward += 10.0 * (1 - features[0])  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Reward based on price change percentage
        else:  # Downtrend
            reward += -features[0] * 20.0  # Reward based on negative price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[4] < 30:  # Oversold condition (RSI)
            reward += 15.0  # Encourage buying
        elif features[4] > 70:  # Overbought condition (RSI)
            reward += -15.0  # Encourage selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))