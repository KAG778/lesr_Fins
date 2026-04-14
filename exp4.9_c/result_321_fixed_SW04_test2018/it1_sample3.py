import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    high_prices = s[2:120:6]      # Extract high prices
    low_prices = s[3:120:6]       # Extract low prices
    volumes = s[4:120:6]          # Extract volumes

    # Feature 1: Price Momentum (percentage change over the last 5 days)
    price_momentum = (closing_prices[0] - closing_prices[5]) / closing_prices[5] if closing_prices[5] != 0 else 0.0

    # Feature 2: Volume Change (percentage change over the last 5 days)
    volume_change = (volumes[0] - volumes[5]) / volumes[5] if volumes[5] != 0 else 0.0

    # Feature 3: Average True Range (ATR) over the last 14 days
    true_ranges = np.maximum(high_prices - low_prices, 
                             np.abs(high_prices - np.roll(closing_prices, 1)),
                             np.abs(low_prices - np.roll(closing_prices, 1)))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0
    
    # Feature 4: Relative Strength Index (RSI) calculation (simplified)
    gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
    losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) > 13 else 0.0
    avg_loss = np.mean(losses[-14:]) if len(losses) > 13 else 0.0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))

    features = [price_momentum, volume_change, atr, rsi]
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
        reward -= 40.0  # Strong negative for BUY-aligned actions
        # Mild positive reward for SELL-aligned actions
        reward += 5.0 if features[0] < 0 else 0  # negative momentum aligns with SELL
    elif risk_level > 0.4:
        reward -= 10.0 if features[0] > 0 else 0  # positive momentum aligns with BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += features[0] * 10.0  # Price momentum aligned with trend direction

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # RSI < 30 indicates oversold
            reward += 5.0  # Positive reward for potential buy signal
        elif features[3] > 70:  # RSI > 70 indicates overbought
            reward += 5.0  # Positive reward for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))