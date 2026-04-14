import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    highs = s[2:120:6]            # Extract high prices
    lows = s[3:120:6]             # Extract low prices

    # Feature 1: Price Change (last day percent change)
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Change (last day percent change)
    avg_volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if volumes[-2] != 0 else 0.0

    # Feature 3: Average True Range (ATR) over the last 20 days
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closing_prices[:-1]), 
                               np.abs(lows[1:] - closing_prices[:-1])))
    atr = np.mean(tr) if len(tr) > 0 else 0.0

    # Feature 4: Price Range normalized by closing price (last day)
    price_range_normalized = (highs[-1] - lows[-1]) / (closing_prices[-1] + 1e-10)

    # Feature 5: Volatility (Standard Deviation of Price Changes over last 20 days)
    price_changes = np.diff(closing_prices)
    volatility = np.std(price_changes[-20:]) if len(price_changes) >= 20 else 0.0

    return np.array([price_change, avg_volume_change, atr, price_range_normalized, volatility])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Extract features for readability
    price_change = features[0]
    avg_volume_change = features[1]
    atr = features[2]
    price_range_normalized = features[3]
    volatility = features[4]

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50.0 * (price_change if price_change > 0 else 1)  # Strong negative for BUY-aligned features
        reward += 10.0 * (avg_volume_change < 0)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 20.0 * (price_change)  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 20.0 * trend_direction * price_change  # Reward for following trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if price_change < -0.01:  # Oversold condition
            reward += 10.0  # Encourage BUY
        elif price_change > 0.01:  # Overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))