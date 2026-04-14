import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract volumes

    if len(closing_prices) < 20 or len(volumes) < 20:
        return np.zeros(6)  # Not enough data to calculate features

    # Feature 1: Volatility (Standard Deviation of closing prices over 14 days)
    volatility = np.std(closing_prices[-14:])

    # Feature 2: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices[-14:] * volumes[-14:]) / np.sum(volumes[-14:]) if np.sum(volumes[-14:]) != 0 else 0

    # Feature 3: Average True Range (ATR)
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]  # Extract low prices
    tr = np.maximum(high_prices[-14:] - low_prices[-14:], 
                    np.maximum(np.abs(high_prices[-14:] - closing_prices[-14:]), 
                               np.abs(low_prices[-14:] - closing_prices[-14:])))
    atr = np.mean(tr)  # Average True Range

    # Feature 4: Price Change Percentage
    price_change_percentage = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 5: Momentum (Current Close - Average Close of the last 5 days)
    momentum = closing_prices[-1] - np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) > 0 else 0

    # Feature 6: Relative Strength Index (RSI)
    deltas = np.diff(closing_prices[-14:])  # Daily price changes
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if (gain + loss) != 0 else 0

    features = [volatility, vwap, atr, price_change_percentage, momentum, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Assuming features are in the context of risk
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        reward += 30 * (1 if trend_direction > 0 else -1)  # Positive reward for momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range