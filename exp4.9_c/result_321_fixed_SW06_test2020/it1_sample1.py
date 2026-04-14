import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) for the last 20 days
    ema = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0

    # Feature 2: Price Change Percentage from the previous closing
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 3: Average True Range (ATR) for volatility estimation over the last 20 days
    true_ranges = np.maximum(0, closing_prices[1:20] - closing_prices[0:19])  # Simplifying for illustration; use full ATR calculation in practice
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0.0

    # Return features
    return np.array([ema, price_change_pct, atr])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        # Mild positive for SELL-aligned features if price is not rising
        reward += 5.0 if features[1] < 0 else 0
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[1] > 0:  # Positive price change percentage
            reward += 10.0 * features[1]  # Reward for momentum alignment
        else:  # Negative price change percentage
            reward += 10.0 * -features[1]  # Penalize for betting against trend

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < -0.01:  # Oversold condition
            reward += 5.0  # Reward for potential buy
        elif features[1] > 0.01:  # Overbought condition
            reward += 5.0  # Reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    # Return reward value clipped to the range [-100, 100]
    return float(np.clip(reward, -100, 100))