import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes
    high_prices = s[2:120:6]      # High prices
    low_prices = s[3:120:6]       # Low prices

    # Feature 1: Price Change Percentage (last day vs previous day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0

    # Feature 3: Price Range (High - Low of the last day)
    price_range = high_prices[-1] - low_prices[-1] if len(high_prices) > 0 and len(low_prices) > 0 else 0.0

    # Feature 4: Exponential Moving Average (EMA) of closing prices (span = 10)
    ema = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0.0

    # Feature 5: Stochastic Oscillator
    min_low = np.min(low_prices[-14:]) if len(low_prices) >= 14 else 0.0
    max_high = np.max(high_prices[-14:]) if len(high_prices) >= 14 else 0.0
    stochastic_oscillator = ((closing_prices[-1] - min_low) / (max_high - min_low)) * 100 if max_high > min_low else 0.0

    features = [price_change_pct, avg_volume, price_range, ema, stochastic_oscillator]
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
        if features[0] < 0:  # If the price change is negative, reward for SELL
            reward += 10.0  # Mild positive for SELL
    elif risk_level > 0.4:
        if features[0] > 0:  # If price change is positive, penalize for BUY
            reward -= 10.0  

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += features[0] * 10.0  # Reward momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.01:  # Oversold condition for buying
            reward += 5.0  # Positive for buying
        elif features[0] > 0.01:  # Overbought condition for selling
            reward += 5.0  # Positive for selling

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))