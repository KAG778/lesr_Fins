import numpy as np

def revise_state(s):
    closing_prices = s[0::6][:20]  # Extract closing prices
    volumes = s[4::6][:20]          # Extract trading volumes

    # Feature 1: Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-6]  # Current vs 5 days ago

    # Feature 2: Bollinger Bands
    sma = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 1e-6
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)
    bb_width = upper_band - lower_band  # Width of the Bollinger Bands

    # Feature 3: Average True Range (ATR)
    high_prices = s[2::6][:20]  # Extract high prices
    low_prices = s[3::6][:20]   # Extract low prices
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(np.abs(high_prices[1:] - closing_prices[:-1][1:]), 
                                         np.abs(low_prices[1:] - closing_prices[:-1][1:])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Combine features into a single array
    features = [price_momentum, bb_width, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Calculate dynamic thresholds using historical std
    historical_std = np.std(features) if np.std(features) != 0 else 1e-6

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Positive price momentum
            reward = np.random.uniform(-50, -30)  # Strong negative for BUY-aligned features
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        if features[0] > 0:
            reward = -10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Bullish trend and momentum
            reward += 20  # Positive reward for correct bullish bet
        elif trend_direction < 0 and features[0] < 0:  # Bearish trend and momentum
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < historical_std:  # Narrow Bollinger Bands indicating potential mean reversion
            reward += 10  # Reward for potential buy
        else:
            reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within valid range
    return np.clip(reward, -100, 100)