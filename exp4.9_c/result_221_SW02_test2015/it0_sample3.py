import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []

    # Ensure at least 20 days of data for moving average calculation
    closing_prices = s[0::6]  # Extracting closing prices
    trading_volumes = s[4::6]  # Extracting trading volumes
    
    # Feature 1: Price Change (percentage change from the previous day)
    price_changes = np.diff(closing_prices) / closing_prices[:-1]
    price_change_feature = np.concatenate(([0], price_changes))  # prepend 0 for alignment
    features.append(price_change_feature[-1])  # last price change

    # Feature 2: Moving Average (5-day simple moving average)
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = closing_prices[-1]  # fallback to last price
    features.append(moving_average)

    # Feature 3: Volume Change (percentage change from the previous day)
    volume_changes = np.diff(trading_volumes) / trading_volumes[:-1]
    volume_change_feature = np.concatenate(([0], volume_changes))  # prepend 0 for alignment
    features.append(volume_change_feature[-1])  # last volume change

    return np.array(features)

def intrinsic_reward(enhanced_state):
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40  # Example strong negative reward for BUY
        # Mild positive reward for SELL-aligned features
        reward += 7  # Example mild positive reward for SELL
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Example moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 15  # Positive reward for strong bullish features
        elif trend_direction < -0.3:
            reward += 15  # Positive reward for strong bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += 10  # Example positive reward for mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)