import numpy as np

def revise_state(s):
    # Extract relevant price and volume data from the raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract volumes

    # Feature 1: Price Momentum (percentage change from the previous day)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Volume Change (percentage change from the previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0.0
    
    # Feature 3: Price Range (high - low over the last day)
    price_range = high_prices[-1] - low_prices[-1] if len(high_prices) > 0 and len(low_prices) > 0 else 0.0

    # Feature 4: Exponential Moving Average (EMA) of closing prices over the last 14 days
    ema_period = 14
    if len(closing_prices) >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])
    else:
        ema = closing_prices[-1] if closing_prices else 0.0

    # Feature 5: Average True Range (ATR) over the last 14 days
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:],
                             np.abs(high_prices[1:] - closing_prices[:-1]),
                             np.abs(low_prices[1:] - closing_prices[:-1]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0

    features = [price_momentum, volume_change, price_range, ema, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Relative thresholds based on historical data (these should be calculated from the dataset)
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    trend_threshold = 0.3

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= 40.0  # Strong negative for BUY-aligned features
        if features[0] < 0:  # Price momentum is negative, mild positive for SELL
            reward += 10.0
    elif risk_level > risk_threshold_medium:
        if features[0] > 0:  # Positive price momentum, penalize for BUY
            reward -= 10.0

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += features[0] * 10.0  # Align reward with price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition for buying
            reward += 5.0  # Positive for potential buying
        elif features[0] > 0.01:  # Overbought condition for selling
            reward += 5.0  # Positive for potential selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude due to high volatility

    return float(np.clip(reward, -100, 100))