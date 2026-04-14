import numpy as np

def revise_state(s):
    # Extract relevant price and volume data from the raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    volumes = s[4:120:6]          # Extract volumes

    # Feature 1: Price Momentum (percentage change from the previous day)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Volume Change (percentage change from the previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0.0
    
    # Feature 3: Price Range (high - low over the last day)
    price_range = high_prices[-1] - low_prices[-1] if len(high_prices) > 0 and len(low_prices) > 0 else 0.0

    # Feature 4: Relative Strength Index (RSI) calculation over the last 14 days
    gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
    losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) > 14 else 0.0
    avg_loss = np.mean(losses[-14:]) if len(losses) > 14 else 0.0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))

    features = [price_momentum, volume_change, price_range, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Relative thresholds based on historical data (example values)
    risk_threshold = 0.5  # Example threshold for high risk
    trend_threshold = 0.3  # Example threshold for trend determination

    # Priority 1: Risk Management
    if risk_level > risk_threshold:
        reward -= 40.0  # Strong negative for BUY-aligned features
        if features[0] < 0:  # If price momentum is negative, reward for SELL
            reward += 10.0  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        if features[0] > 0:  # If price momentum is positive, penalize for BUY
            reward -= 10.0  

    # Priority 2: Trend Following
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        reward += features[0] * 10.0  # Reward momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[3] < 30:  # RSI < 30 indicates oversold
            reward += 5.0  # Positive for potential buying
        elif features[3] > 70:  # RSI > 70 indicates overbought
            reward += 5.0  # Positive for potential selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude due to high volatility

    return float(np.clip(reward, -100, 100))