import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)

    # Initialize an empty list to store new features
    features = []

    # Calculate the daily returns
    closes = s[0::6]  # closing prices
    daily_returns = np.diff(closes) / closes[:-1]  # daily returns
    features.append(np.mean(daily_returns) if len(daily_returns) > 0 else 0)  # Mean daily return

    # Calculate volatility as the standard deviation of returns
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
    features.append(volatility)

    # Compute Relative Strength Index (RSI) for the last 14 days
    window_length = 14
    if len(daily_returns) > window_length:
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)

        avg_gain = np.mean(gains[-window_length:])
        avg_loss = np.mean(losses[-window_length:])

        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)
    else:
        features.append(50)  # Neutral RSI

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)
        # MILD positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10)
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(5, 15)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for bullish features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold→buy, overbought→sell)
        reward += 10  # Example positive reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Constrain reward within [-100, 100] if necessary
    reward = max(min(reward, 100), -100)

    return reward