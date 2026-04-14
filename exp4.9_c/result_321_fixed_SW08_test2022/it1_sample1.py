import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    volumes = s[4:120:6]  # Trading volumes for 20 days
    high_prices = s[2:120:6]  # High prices
    low_prices = s[3:120:6]  # Low prices

    # Feature 1: Price Momentum (current closing price - previous closing price)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0
    
    # Feature 2: Average Volume Over Last 20 Days
    average_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Price Range (high - low)
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 and len(low_prices) > 0 else 0

    # Feature 4: Standard Deviation of Closing Prices
    price_std = np.std(closing_prices) if len(closing_prices) > 0 else 0

    # Feature 5: Relative Strength Index (RSI) for recent momentum (using a simple 14-day RSI calculation)
    if len(closing_prices) >= 14:
        delta = np.diff(closing_prices[-14:])  # Price changes over the last 14 days
        gain = np.sum(delta[delta > 0]) / 14
        loss = -np.sum(delta[delta < 0]) / 14
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI if not enough data

    features = [price_momentum, average_volume, price_range, price_std, rsi]
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
        reward += 10.0 if features[0] < 0 else 0  # Mild positive for SELL-aligned features if price momentum is negative
    elif risk_level > 0.4:
        reward -= 20.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Price momentum positively aligned
        else:  # Downtrend
            reward += -features[0] * 20.0  # Price momentum negatively aligned

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[4] < 30:  # Assuming RSI < 30 indicates oversold condition
            reward += 15.0  # Buy signal
        elif features[4] > 70:  # Assuming RSI > 70 indicates overbought condition
            reward += -10.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))