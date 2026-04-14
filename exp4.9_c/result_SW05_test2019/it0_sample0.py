import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]  # Extract low prices
    volumes = s[4::6]  # Extract volumes

    # Feature 1: Price Momentum (last closing - first closing)
    price_momentum = closing_prices[-1] - closing_prices[0]

    # Feature 2: MACD
    # For simplicity, we use a fast (12 days) and slow (26 days) average
    short_window = 12
    long_window = 26
    if len(closing_prices) >= long_window:
        short_ema = np.mean(closing_prices[-short_window:])  # Simplified EMA
        long_ema = np.mean(closing_prices[-long_window:])  # Simplified EMA
        macd = short_ema - long_ema
    else:
        macd = 0.0

    # Feature 3: RSI
    delta = np.diff(closing_prices)
    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
    loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
    rs = gain / loss if loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    features = [price_momentum, macd, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY
        reward += 10   # MILD POSITIVE reward for SELL
    elif risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive reward for BUY-aligned features
        else:
            reward += 10  # Positive reward for SELL-aligned features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)