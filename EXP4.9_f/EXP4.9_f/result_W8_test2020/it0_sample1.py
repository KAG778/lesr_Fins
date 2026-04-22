import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extracting closing prices
    high_prices = s[2::6]      # Extracting high prices
    low_prices = s[3::6]       # Extracting low prices
    volumes = s[4::6]          # Extracting trading volumes

    # Feature 1: Price Momentum - difference between the most recent closing price and the closing price 5 days ago
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average True Range (ATR)
    # ATR = (High - Low) for each day, averaged over the last 14 days
    true_ranges = high_prices - low_prices
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Feature 3: Relative Strength Index (RSI) - typically calculated over 14 periods
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [price_momentum, atr, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY signals
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified bounds