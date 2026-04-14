import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    N = len(closing_prices)

    # Feature 1: 10-Day Exponential Moving Average (EMA)
    ema_period = 10
    if N >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Simplified EMA for demonstration
    else:
        ema = np.nan

    # Feature 2: Rate of Change (ROC)
    roc_period = 5
    if N > roc_period:
        roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]  # Rate of Change
    else:
        roc = np.nan

    # Feature 3: Average True Range (ATR) for volatility
    if N > 1:
        true_ranges = np.maximum(
            np.diff(s[2::6]),  # High - Low
            np.maximum(
                np.abs(np.diff(s[1::6])),  # High - Previous Close
                np.abs(np.diff(s[2::6]))   # Low - Previous Close
            )
        )
        atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.nan
    else:
        atr = np.nan

    # Feature 4: Volume Weighted Average Price (VWAP)
    if np.sum(volumes) > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes)  # VWAP calculation
    else:
        vwap = np.nan

    # Feature 5: Drawdown from the highest price in the last 20 days
    if N >= 20:
        max_price = np.max(closing_prices[-20:])
        current_price = closing_prices[-1]
        drawdown = (max_price - current_price) / max_price
    else:
        drawdown = np.nan

    # Combine features into a single array
    features = [ema, roc, atr, vwap, drawdown]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on the historical standard deviation of features
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s) > 123 else 1  # Avoid division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for high risk when BUY
        return np.clip(reward, -100, 100)  # Early return
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Bullish Trend
            reward += 30  # Strong positive reward for bullish trend
        else:  # Bearish Trend
            reward += 30  # Strong positive reward for bearish trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std and risk_level < 0.4:  # Relative to historical volatility
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within bounds
    return np.clip(reward, -100, 100)