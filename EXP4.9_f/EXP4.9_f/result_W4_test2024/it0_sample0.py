import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    closing_prices = s[0:120:6]  # Extract closing prices (s[i*6 + 0] for i=0..19)
    days = len(closing_prices)

    # Feature 1: Price Momentum
    price_momentum = closing_prices[-1] - np.mean(closing_prices) if days > 0 else 0

    # Feature 2: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0  # Last 14 days
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0  # Last 14 days

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: EMA Difference (short term vs long term)
    short_window = 5
    long_window = 20
    
    if days >= long_window:
        short_ema = np.mean(closing_prices[-short_window:])  # Short term EMA
        long_ema = np.mean(closing_prices[-long_window:])    # Long term EMA
        ema_difference = short_ema - long_ema
    else:
        ema_difference = 0

    features = [price_momentum, rsi, ema_difference]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward += -40  # Example value
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example value

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # Positive reward for buy aligned features in uptrend
        else:
            reward += 20  # Positive reward for sell aligned features in downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Here we would reward mean-reversion features
        reward += 10  # Example reward for mean-reversion scenario

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]