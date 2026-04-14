import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    
    # Edge case handling
    if len(closing_prices) < 2:
        return np.zeros(4)  # Return zeros if there are not enough days of data

    # Feature 1: Volatility-Adjusted Momentum
    momentum = closing_prices[-1] - closing_prices[-2]
    historical_volatility = np.std(closing_prices[-14:]) if len(closing_prices) >= 14 else 1  # Avoid division by zero
    volatility_adjusted_momentum = momentum / historical_volatility

    # Feature 2: Exponential Moving Average (EMA) Crossover
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else np.mean(closing_prices)
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else np.mean(closing_prices)
    ema_crossover = short_ema - long_ema  # Positive if short EMA is above long EMA

    # Feature 3: Relative Strength Index (RSI) Variation
    gains = np.where(np.diff(closing_prices[-14:]) > 0, np.diff(closing_prices[-14:]), 0)
    losses = -np.where(np.diff(closing_prices[-14:]) < 0, np.diff(closing_prices[-14:]), 0)
    average_gain = np.mean(gains) if len(gains) > 0 else 0
    average_loss = np.mean(losses) if len(losses) > 0 else 0
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))  # Standard RSI calculation

    # Feature 4: Volume Change Percentage
    if len(s) >= 6:
        recent_volume = s[4::6][-1]
        previous_volume = s[4::6][-2]
        volume_change_percentage = ((recent_volume - previous_volume) / previous_volume) * 100 if previous_volume != 0 else 0
    else:
        volume_change_percentage = 0

    features = [volatility_adjusted_momentum, ema_crossover, rsi, volume_change_percentage]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical volatility
    risk_threshold_high = 0.7 * np.std(enhanced_s[123:])  # High risk threshold
    risk_threshold_moderate = 0.4 * np.std(enhanced_s[123:])  # Moderate risk threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -50  # Strong negative reward for BUY signals
        reward += 10   # Mild positive reward for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 30  # Positive reward for buying in an uptrend
        else:  # Downtrend
            reward += 30  # Positive reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is capped within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward