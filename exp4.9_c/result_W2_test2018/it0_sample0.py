import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[0::6]
    opening_prices = s[1::6]
    high_prices = s[2::6]
    low_prices = s[3::6]
    volumes = s[4::6]

    features = []

    # Feature 1: Price Momentum (Rate of Change)
    # Calculating momentum as the difference between the latest closing price and the price N days ago
    momentum_period = 5  # Example momentum period
    if len(closing_prices) > momentum_period:
        momentum = (closing_prices[-1] - closing_prices[-(momentum_period + 1)]) / closing_prices[-(momentum_period + 1)]
    else:
        momentum = 0.0  # Handle edge case
    features.append(momentum)

    # Feature 2: Moving Average Convergence Divergence (MACD)
    # Calculate the short-term and long-term EMA
    short_ema_period = 12
    long_ema_period = 26
    short_ema = np.mean(closing_prices[-short_ema_period:]) if len(closing_prices) >= short_ema_period else 0
    long_ema = np.mean(closing_prices[-long_ema_period:]) if len(closing_prices) >= long_ema_period else 0
    macd = short_ema - long_ema
    features.append(macd)

    # Feature 3: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], high_prices[1:] - closing_prices[:-1], closing_prices[:-1] - low_prices[1:])
    atr_period = 14
    atr = np.mean(tr[-atr_period:]) if len(tr) >= atr_period else 0  # Handle edge case
    features.append(atr)

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
        if enhanced_s[123] > 0:  # Assuming positive feature indicates a BUY signal
            reward = -50  # Strong negative for BUY-aligned features
        else:
            reward = 10  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        if enhanced_s[123] > 0:  # Assuming positive feature indicates a BUY signal
            reward = -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # Positive feature and uptrend
            reward = 20  # Positive reward for correct bullish signal
        elif trend_direction < -0.3 and enhanced_s[123] < 0:  # Negative feature and downtrend
            reward = 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] > 0:  # Assuming positive feature indicates a BUY signal
            reward = 10  # Reward mean-reversion features
        else:
            reward = -10  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)