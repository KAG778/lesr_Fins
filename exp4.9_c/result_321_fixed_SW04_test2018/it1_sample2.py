import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    high_prices = s[2:120:6]      # Extracting high prices
    low_prices = s[3:120:6]       # Extracting low prices
    volumes = s[4:120:6]          # Extracting volumes

    # Feature 1: Price Change Rate (percentage change from the last close)
    price_change_rate = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume Change (percentage change from the previous day)
    avg_volume_change = np.mean(np.diff(volumes) / volumes[:-1]) if len(volumes) > 1 and np.all(volumes[:-1] != 0) else 0.0

    # Feature 3: Price Range over last 20 days
    price_range = np.max(high_prices) - np.min(low_prices) if len(high_prices) > 0 else 0.0

    # Feature 4: Exponential Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:])  # Short EMA (12-day)
    long_ema = np.mean(closing_prices[-26:])   # Long EMA (26-day)
    macd = short_ema - long_ema

    features = [price_change_rate, avg_volume_change, price_range, macd]
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
        # Mild positive for SELL if price change rate indicates downward movement
        reward += 10.0 if features[0] < 0 else 0.0  # BUY is risky, so reward SELL if price is falling
    elif risk_level > 0.4:
        reward -= 10.0 if features[0] > 0 else 0.0  # Penalty for BUY if price change rate is positive

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += features[0] * 15.0 * trend_direction  # Align reward with trend direction

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition for buying
            reward += 10.0  # Positive for potential buying
        elif features[0] > 0.01:  # Overbought condition for selling
            reward += 10.0  # Positive for potential selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude due to high volatility

    return float(np.clip(reward, -100, 100))