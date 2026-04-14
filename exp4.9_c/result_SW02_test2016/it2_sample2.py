import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract volumes

    if len(closing_prices) < 20 or len(volumes) < 20:
        return np.zeros(5)  # Return zeros if there are not enough data points

    # Feature 1: Adjusted Momentum
    momentum = closing_prices[-1] - closing_prices[-2]
    historical_volatility = np.std(closing_prices[-14:]) if len(closing_prices) >= 14 else 1
    adjusted_momentum = momentum / historical_volatility  # Normalize momentum by volatility

    # Feature 2: Bollinger Band Width
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
    bollinger_band_width = (std_dev / sma) * 100 if sma != 0 else 0  # Percentage width

    # Feature 3: Rate of Change (ROC)
    roc = ((closing_prices[-1] - closing_prices[-10]) / closing_prices[-10]) * 100 if closing_prices[-10] != 0 else 0  # 10-day ROC

    # Feature 4: Volume-Weighted Momentum
    vwap = np.sum(closing_prices[-14:] * volumes[-14:]) / np.sum(volumes[-14:]) if np.sum(volumes[-14:]) != 0 else 0
    volume_weighted_momentum = (closing_prices[-1] - vwap) / vwap if vwap != 0 else 0  # Compare to VWAP

    # Feature 5: Relative Strength Index (RSI) Change
    deltas = np.diff(closing_prices[-14:])  # Daily price changes
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))  # Standard RSI calculation
    rsi_change = rsi - 50  # Normalize RSI to center around zero

    features = [adjusted_momentum, bollinger_band_width, roc, volume_weighted_momentum, rsi_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Calculate relative thresholds based on historical volatility
    historical_std = np.std(enhanced_s[123:])
    risk_threshold_high = 0.7 * historical_std  # High risk threshold
    risk_threshold_moderate = 0.4 * historical_std  # Moderate risk threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 30  # Positive reward for buying in an uptrend
        else:  # Downtrend
            reward += 30  # Positive reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range