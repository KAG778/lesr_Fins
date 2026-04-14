import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Get closing prices (day i at i*6)
    features = []
    
    # 1. Compute daily price change (percentage)
    price_changes = np.diff(closing_prices) / closing_prices[:-1]  # Percentage change
    # Append the last day change (0 if not available)
    features.append(price_changes[-1] if len(price_changes) > 0 else 0)

    # 2. Compute 5-day moving average (simple)
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])  # Last 5 days
    else:
        moving_average = closing_prices[-1]  # Fallback to last closing price
    features.append(moving_average)

    # 3. Compute RSI (14-day)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        average_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        average_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        if average_loss == 0:
            return 100  # Avoid division by zero
        rs = average_gain / average_loss
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)
    features.append(rsi)

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
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE for BUY-aligned features
        reward += +7   # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive for upward features
        else:
            reward += 10  # Positive for downward features (correct bearish bet)

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features
        reward -= 5  # Penalize breakout-chasing features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds