import numpy as np

def revise_state(s):
    # s: 120d raw state
    # We will be calculating three new features:
    # 1. Price Momentum
    # 2. Relative Strength Index (RSI)
    # 3. Moving Average Convergence Divergence (MACD)

    closing_prices = s[0:120:6]  # Extract closing prices
    recent_close = closing_prices[-1]
    historical_close = closing_prices[-6] if len(closing_prices) > 5 else recent_close  # Previous close for momentum

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = recent_close - historical_close

    # Feature 2: Relative Strength Index (RSI)
    # Calculate RSI over the last 14 days if possible
    gains = []
    losses = []
    for i in range(len(closing_prices) - 1):
        change = closing_prices[i + 1] - closing_prices[i]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)

    average_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
    average_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)

    # Avoid division by zero
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: MACD - 12-day EMA and 26-day EMA
    # Using a simple EMA calculation for MACD
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else np.mean(closing_prices)
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else np.mean(closing_prices)
    macd = short_ema - long_ema

    features = [price_momentum, rsi, macd]
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
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        reward += 5    # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # positive reward for upward features
        else:
            reward += 20  # positive reward for downward features (correct bearish bet)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # rewarding mean-reversion features (oversold→buy, overbought→sell)
        reward += -10  # penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)