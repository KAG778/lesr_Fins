import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Moving Average (last 5 days)
    ma_window = 5
    moving_average = np.mean(closing_prices[-ma_window:]) if len(closing_prices) >= ma_window else 0

    # Feature 2: Price Momentum (current close - close 2 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-3] if len(closing_prices) >= 3 else 0

    # Feature 3: Volume Change (current volume - average volume of last 5 days)
    avg_volume = np.mean(volumes[-ma_window:]) if len(volumes) >= ma_window else 0
    volume_change = volumes[-1] - avg_volume if avg_volume > 0 else 0

    features = [moving_average, price_momentum, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if len(features) > 0:
            reward += trend_direction * features[0] * 10.0  # Using moving average for trend following

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming features[1] is used for mean reversion (price momentum)
        if features[1] < 0:  # Oversold condition
            reward += 5.0  # Positive for oversold
        elif features[1] > 0:  # Overbought condition
            reward -= 5.0  # Negative for overbought

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))