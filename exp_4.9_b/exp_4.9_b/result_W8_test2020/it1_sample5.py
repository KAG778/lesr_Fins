import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema[-1]

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if np.any(gain[-window:]) else 0
    avg_loss = np.mean(loss[-window:]) if np.any(loss[-window:]) else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range."""
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.abs(highs[-window:] - closes[-window:]), 
                    np.abs(lows[-window:] - closes[-window:]))
    return np.mean(tr)

def calculate_volatility(returns, window):
    """Calculate historical volatility."""
    if len(returns) < window:
        return np.nan
    return np.std(returns[-window:]) * 100

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)

    # Volatility Indicators
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_vol_5 = calculate_volatility(daily_returns, 5)
    historical_vol_20 = calculate_volatility(daily_returns, 20)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume Indicators
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    volume_avg_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.nan
    volume_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Market Regime Features
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 > 0 else np.nan
    trend_strength = np.corrcoef(closing_prices[-20:], range(len(closing_prices[-20:])))[0, 1] if len(closing_prices[-20:]) > 1 else np.nan

    # Collecting features
    enhanced_s = np.concatenate((
        s,
        [sma_5, sma_10, sma_20, ema_5, ema_10,
         rsi_5, rsi_14, historical_vol_5, historical_vol_20, atr_14,
         obv[-1], volume_avg_5, volume_avg_20, price_position,
         volatility_ratio, trend_strength, volume_ratio]
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Position flag (1.0 = holding, 0.0 = not holding)
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    recent_return = returns[-1] if len(returns) > 0 else 0

    # Calculate historical volatility for adaptive threshold
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Sharp decline
            reward -= 50
    else:  # Holding
        if recent_return > threshold:  # Continue to hold in uptrend
            reward += 25
        elif recent_return < -threshold:  # Trend weakening; consider selling
            reward -= 50

    # Penalize uncertain/choppy market conditions
    if np.abs(recent_return) < (0.5 * threshold):  # Choppy condition
        reward -= 10

    # Clip reward to ensure it is within bounds
    return np.clip(reward, -100, 100)