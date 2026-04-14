import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    highs = s[2:120:6]            # Extract high prices
    lows = s[3:120:6]             # Extract low prices
    
    # Feature 1: Price Change (% Change from previous closing price)
    price_changes = np.diff(closing_prices) / closing_prices[:-1]
    avg_price_change = np.mean(price_changes) if len(price_changes) > 0 else 0.0

    # Feature 2: Average Volume Change
    volume_changes = np.diff(volumes) / volumes[:-1]
    avg_volume_change = np.mean(volume_changes) if len(volume_changes) > 0 else 0.0

    # Feature 3: Average True Range (ATR) over the last 20 days
    atr = np.mean(np.maximum(highs[1:] - lows[1:], highs[:-1] - closing_prices[:-1], closing_prices[:-1] - lows[:-1])) if len(highs) > 1 else 0.0

    # Feature 4: Price Range normalized by closing price
    price_range_normalized = (np.max(highs) - np.min(lows)) / (np.mean(closing_prices) + 1e-10)

    # Return computed features as a numpy array
    return np.array([avg_price_change, avg_volume_change, atr, price_range_normalized])

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
        reward += 10.0  # Mild positive for SELL-aligned features (risk-off)
    elif risk_level > 0.4:
        reward -= 15.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 15.0  # Align reward with trend direction and price change

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition
            reward += 10.0  # Encourage BUY
        elif features[0] > 0.01:  # Overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))