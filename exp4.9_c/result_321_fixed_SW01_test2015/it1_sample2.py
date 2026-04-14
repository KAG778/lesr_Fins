import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    
    num_days = len(closing_prices)
    
    # Feature 1: Exponential Moving Average (EMA) over the last 5 days
    if num_days >= 5:
        ema = np.mean(closing_prices[-5:])  # Simple approximation of EMA
    else:
        ema = closing_prices[-1] if num_days > 0 else 0.0
    
    # Feature 2: Average True Range (ATR) for volatility
    true_ranges = np.maximum(
        np.maximum(closing_prices[1:] - closing_prices[:-1], 
                   closing_prices[:-1] - closing_prices[1:]),
        np.abs(closing_prices[1:] - closing_prices[:-1])
    )
    
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0  # ATR over the last 14 days

    # Feature 3: Cumulative Return over the last 10 days
    cumulative_return = (closing_prices[-1] / closing_prices[-11] - 1) * 100 if num_days >= 10 else 0.0
    
    features = [ema, atr, cumulative_return]
    
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
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 5.0 * abs(features[0])  # Mild positive for SELL-aligned features based on EMA
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # EMA affecting reward direction

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0:  # Cumulative return negative indicates potential mean reversion
            reward += 10.0  # Positive for potential buy signal
        elif features[2] > 0:  # Cumulative return positive indicates potential overbought
            reward -= 10.0  # Negative for potential sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))