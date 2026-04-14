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

    # Feature 2: Rate of Change (ROC) over the last 5 days
    roc_period = 5
    if N > roc_period:
        roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]  # Rate of Change
    else:
        roc = np.nan

    # Feature 3: Average True Range (ATR) over the last 14 days for volatility
    atr_period = 14
    if N >= atr_period:
        high = np.array(s[2::6][-atr_period:])  # High prices
        low = np.array(s[1::6][-atr_period:])  # Low prices
        close = np.array(s[0::6][-atr_period:])  # Close prices
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr = np.mean(tr) if len(tr) > 0 else np.nan
    else:
        atr = np.nan

    # Feature 4: Volume Weighted Average Price (VWAP)
    if np.sum(volumes) > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes)
    else:
        vwap = np.nan

    # Feature 5: Drawdown from the highest price in the last 20 days
    if N >= 20:
        max_price = np.max(closing_prices[-20:])
        current_price = closing_prices[-1]
        drawdown = (max_price - current_price) / max_price if max_price > 0 else np.nan
    else:
        drawdown = np.nan

    features = [ema, roc, atr, vwap, drawdown]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Calculate relative thresholds based on historical standard deviation of features
    historical_std = np.std(enhanced_s[123:])  # Use features starting from index 123
    risk_threshold_high = historical_std * 1.5  # 1.5 times the historical std as high risk
    risk_threshold_mid = historical_std * 1.0  # 1.0 times the historical std as mid risk
    trend_threshold = 0.3  # Using absolute trend threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for high risk when BUY
        return np.clip(reward, -100, 100)  # Early return
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if np.abs(trend_direction) > trend_threshold and risk_level < risk_threshold_mid:
        if trend_direction > 0:  # Bullish trend
            reward += 30  # Strong positive reward for alignment with upward trend
        else:  # Bearish trend
            reward += 30  # Strong positive reward for alignment with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if np.abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)