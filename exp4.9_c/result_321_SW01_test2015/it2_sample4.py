import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]  # Trading volumes
    N = len(closing_prices)

    # Feature 1: Exponential Moving Average (EMA) over the last 10 days
    ema_period = 10
    if N >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Simplified EMA for demonstration
    else:
        ema = np.nan

    # Feature 2: Average True Range (ATR) for volatility measurement
    def calculate_atr(prices, period=14):
        if len(prices) < period:
            return np.nan
        high = prices[2::6]  # High prices
        low = prices[3::6]   # Low prices
        close = prices[0::6]  # Close prices
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.nan
        return atr

    atr = calculate_atr(s)

    # Feature 3: Drawdown from the highest price in the last 20 days
    if N >= 20:
        max_price = np.max(closing_prices[-20:])
        current_price = closing_prices[-1]
        drawdown = (max_price - current_price) / max_price
    else:
        drawdown = np.nan

    # Feature 4: Rate of Change (ROC)
    roc_period = 5
    if N > roc_period:
        roc = (closing_prices[-1] - closing_prices[-roc_period]) / closing_prices[-roc_period]  # Rate of Change
    else:
        roc = np.nan

    # Feature 5: Volume Weighted Average Price (VWAP)
    if N > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else np.nan
    else:
        vwap = np.nan

    # Combine features into a single array
    features = [ema, atr, drawdown, roc, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(enhanced_s[123:]) if len(enhanced_s) > 123 else 1.0  # Default to 1.0 to avoid division by zero

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for BUY-aligned features
        return np.clip(reward, -100, 100)  # Early return if risk is high
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 30  # Positive reward for bullish alignment
        else:  # Downtrend
            reward += 30  # Positive reward for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_std and risk_level < 0.4:  # Using relative threshold
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)