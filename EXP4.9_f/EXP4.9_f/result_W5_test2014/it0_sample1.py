import numpy as np

def revise_state(s):
    # s: 120d raw state (OHLCV interleaved)
    closing_prices = s[::6]  # Extracting closing prices
    volumes = s[4::6]        # Extracting volumes

    # 1. Price Momentum (current closing price - closing price 5 days ago)
    if len(closing_prices) > 5:
        price_momentum = closing_prices[0] - closing_prices[5]
    else:
        price_momentum = 0.0  # Edge case handling

    # 2. Volume Momentum (current volume - average volume over the last 5 days)
    if len(volumes) > 5:
        avg_volume = np.mean(volumes[1:6])  # Average of the last 5 days' volumes
        volume_momentum = volumes[0] - avg_volume
    else:
        volume_momentum = 0.0  # Edge case handling

    # 3. Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 50.0  # Neutral RSI if not enough data
        deltas = np.diff(prices)
        gain = np.sum(np.where(deltas > 0, deltas, 0)) / period
        loss = -np.sum(np.where(deltas < 0, deltas, 0)) / period
        rs = gain / loss if loss != 0 else 0  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices)

    features = [price_momentum, volume_momentum, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = computed features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= 40  # Strong negative reward
        # MILD POSITIVE reward for SELL-aligned features
        reward += 7   # Mild positive reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 15

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Reward for upward features
            reward += 10  # Positive reward
        elif trend_direction < -0.3:
            # Reward for downward features
            reward += 10  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        # Penalize breakout-chasing features
        reward += 5  # Positive reward for mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))