import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices

    # Feature 1: Recent Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Over Last 20 Days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Price Range (high - low) over the last 20 days
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 and len(low_prices) > 0 else 0.0

    # Feature 4: Historical Volatility (standard deviation of closing prices)
    historical_volatility = np.std(closing_prices) if len(closing_prices) > 0 else 0.0

    # Feature 5: Momentum Indicator (current closing price vs moving average)
    moving_average_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    momentum = closing_prices[-1] - moving_average_5

    # Feature 6: Relative Strength Index (RSI) for recent momentum
    if len(closing_prices) >= 14:
        delta = np.diff(closing_prices[-14:])  # Price changes over the last 14 days
        gain = np.sum(delta[delta > 0]) / 14
        loss = -np.sum(delta[delta < 0]) / 14
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if not enough data

    features = [price_change_pct, average_volume, price_range, historical_volatility, momentum, rsi]
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
        reward -= 50.0  # Strong negative for buying in high risk
        reward += 10.0 * (1 - features[0])  # Mild positive for sell signal if price change is negative
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for buying in elevated risk

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += features[0] * 20.0 if trend_direction > 0 else -features[0] * 20.0  # Align reward with trend direction

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[5] < 30:  # Assuming RSI < 30 indicates oversold condition
            reward += 15.0  # Encourage buying
        elif features[5] > 70:  # Assuming RSI > 70 indicates overbought condition
            reward -= 10.0  # Encourage selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))