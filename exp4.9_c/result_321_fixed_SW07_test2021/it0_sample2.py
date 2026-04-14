import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    closing_prices = s[0::6]  # Extracting closing prices
    opening_prices = s[1::6]  # Extracting opening prices
    high_prices = s[2::6]     # Extracting high prices
    low_prices = s[3::6]      # Extracting low prices
    volumes = s[4::6]         # Extracting volumes

    # Feature 1: Price Momentum (Current closing price - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Average Trading Volume
    avg_volume = np.mean(volumes) if np.sum(volumes) > 0 else 0

    # Feature 3: Price Range (Highest price - Lowest price over the last 20 days)
    price_range = np.max(high_prices) - np.min(low_prices)

    features = [price_momentum, avg_volume, price_range]
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

    # Priority 1: Risk Management (deterministic, NO random calls)
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 7.0 * features[0]  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10.0 * features[0]  # Positive reward for momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 10.0 * -features[0]  # Positive reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Buy signal reward
        elif features[0] > 0:  # Overbought condition
            reward += -5.0  # Sell signal penalty

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))