import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices every 6th index starting from 0
    opening_prices = s[1::6]   # Opening prices every 6th index starting from 1
    high_prices = s[2::6]      # High prices every 6th index starting from 2
    low_prices = s[3::6]       # Low prices every 6th index starting from 3
    volumes = s[4::6]          # Trading volumes every 6th index starting from 4
    adjusted_closing_prices = s[5::6]  # Adjusted closing prices every 6th index starting from 5

    # Feature 1: Price Change Percentage (last day vs previous day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Average Volume over the last 5 days (to capture volume trends)
    avg_volume = np.mean(volumes[-5:]) if len(volumes[-5:]) > 0 else 0

    # Feature 3: Price Range (High - Low) of the last day
    price_range = high_prices[-1] - low_prices[-1] if high_prices[-1] != low_prices[-1] else 0

    features = [price_change_pct, avg_volume, price_range]
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
        # Optionally, assign positive reward for SELL-aligned features:
        reward += 7.0 * features[0]  # Positive reward for selling during high risk
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += features[0] * 20.0  # Positive reward for upward momentum
        else:  # Downtrend
            reward += -features[0] * 20.0  # Positive reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Oversold condition
            reward += 15.0  # Buy signal
        elif features[0] > 0.05:  # Overbought condition
            reward += 15.0  # Sell signal
        else:
            reward -= 5.0  # Penalize for breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))