import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    opening_prices = s[1:120:6]  # Extract opening prices
    high_prices = s[2:120:6]     # Extract high prices
    low_prices = s[3:120:6]      # Extract low prices
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Price Momentum (current closing price vs opening price)
    price_momentum = (closing_prices[-1] - opening_prices[-1]) / (opening_prices[-1] + 1e-10)

    # Feature 2: Relative Strength Index (RSI) (simplified version for last 14 days)
    gains = np.where(np.diff(closing_prices) > 0, np.diff(closing_prices), 0)
    losses = np.where(np.diff(closing_prices) < 0, -np.diff(closing_prices), 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: Average True Range (ATR) over the last 20 days
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-20:]) if len(tr) > 20 else 0.0

    # Feature 4: Volume Trend (current volume vs. average volume of the last 20 days)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1e-10  # Avoid division by zero
    volume_trend = (volumes[-1] - avg_volume) / avg_volume

    return np.array([price_momentum, rsi, atr, volume_trend])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0
    price_momentum = features[0]
    rsi = features[1]
    atr = features[2]
    volume_trend = features[3]

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0 * (price_momentum if price_momentum > 0 else 1)  # Strong negative for BUY-aligned features
        reward += 5.0 * (volume_trend < 0)  # Mild positive for SELL-aligned features (less volume)
    elif risk_level > 0.4:
        reward -= 10.0 * price_momentum  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * price_momentum * 20.0  # Reward for following trend based on price momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi < 30:  # Oversold condition
            reward += 10.0  # Encourage BUY
        elif rsi > 70:  # Overbought condition
            reward += 10.0  # Encourage SELL

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))