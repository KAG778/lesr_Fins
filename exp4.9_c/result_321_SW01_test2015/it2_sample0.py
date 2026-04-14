import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    N = len(closing_prices)

    features = []

    # Feature 1: 10-Day Exponential Moving Average (EMA)
    ema_period = 10
    if N >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Using simple average as a proxy for EMA
    else:
        ema = np.nan
    features.append(ema)

    # Feature 2: Rate of Change (ROC) over the last 5 days
    roc_period = 5
    if N > roc_period:
        roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]  # Rate of Change
    else:
        roc = np.nan
    features.append(roc)

    # Feature 3: Average True Range (ATR) for volatility measurement
    def calculate_atr(prices, period=14):
        if len(prices) < period:
            return np.nan
        high = prices[1::6]
        low = prices[2::6]
        close = prices[0::6]
        true_ranges = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr = np.mean(true_ranges[-period:]) if len(true_ranges) >= period else np.nan
        return atr

    atr = calculate_atr(s)
    features.append(atr)

    # Feature 4: Drawdown from the highest price in the last 20 days
    if N >= 20:
        max_price = np.max(closing_prices[-20:])
        current_price = closing_prices[-1]
        drawdown = (max_price - current_price) / max_price
    else:
        drawdown = np.nan
    features.append(drawdown)

    # Feature 5: Volume Weighted Average Price (VWAP)
    if N > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else np.nan
    else:
        vwap = np.nan
    features.append(vwap)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical std deviation
    historical_features = enhanced_s[123:]  # Use the features to calculate thresholds
    historical_std = np.std(historical_features)

    risk_threshold_high = historical_std * 1.5
    risk_threshold_mid = historical_std * 1.2
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for high risk when BUY
        return max(-100, min(100, reward))  # Early return
    elif risk_level > risk_threshold_mid:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_mid:
        if trend_direction > 0:  # Uptrend
            reward += 30  # Strong positive reward for bullish momentum
        else:  # Downtrend
            reward += 30  # Strong positive reward for bearish momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.std(historical_features) and risk_level < risk_threshold_mid:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return max(-100, min(100, reward))