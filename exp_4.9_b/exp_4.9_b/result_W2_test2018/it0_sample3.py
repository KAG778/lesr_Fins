import numpy as np

def calculate_sma(prices, period):
    if len(prices) < period:
        return np.nan
    return np.mean(prices[-period:])

def calculate_ema(prices, period):
    if len(prices) < period:
        return np.nan
    ema = prices[-period]
    multiplier = 2 / (period + 1)
    for price in prices[-period+1:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_rsi(prices, period):
    if len(prices) < period:
        return np.nan
    
    deltas = np.diff(prices[-period:])
    gains = np.where(deltas > 0, deltas, 0).mean()
    losses = np.abs(np.where(deltas < 0, deltas, 0)).mean()
    
    if losses == 0:
        return 100  # If no losses, RSI is 100
    rs = gains / losses
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd_line = ema_12 - ema_26
    signal_line = calculate_ema(prices[-26:], 9) if len(prices) >= 26 else np.nan
    histogram = macd_line - signal_line if not np.isnan(macd_line) and not np.isnan(signal_line) else np.nan
    return macd_line, signal_line, histogram

def calculate_atr(highs, lows, closes, period):
    if len(highs) < period:
        return np.nan
    tr = np.maximum(highs[-period:] - lows[-period:], np.maximum(np.abs(highs[-period:] - closes[-period]), np.abs(lows[-period:] - closes[-period])))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    price_vs_sma_5 = closing_prices[-1] / sma_5 if sma_5 else np.nan
    price_vs_ema_10 = closing_prices[-1] / ema_10 if ema_10 else np.nan

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    macd_line, signal_line, macd_histogram = calculate_macd(closing_prices)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) > 1 else np.nan

    # Volatility Indicators
    hist_vol_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 6 else np.nan
    hist_vol_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 21 else np.nan
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan

    # Market Regime Detection
    volatility_ratio = hist_vol_5 / hist_vol_20 if hist_vol_20 > 0 else np.nan
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]**2 if len(closing_prices) > 1 else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan
    
    # Create enhanced state
    enhanced_s = np.array([sma_5, sma_10, sma_20, ema_5, ema_10,
                           price_vs_sma_5, price_vs_ema_10,
                           rsi_5, rsi_14,
                           macd_line, signal_line, macd_histogram, momentum,
                           hist_vol_5, hist_vol_20, atr,
                           obv, volume_ratio,
                           volatility_ratio, price_position, trend_strength, volume_ratio_regime])

    return np.concatenate([s, enhanced_s])

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    if len(closing_prices) < 2:
        return 0  # Not enough data to calculate reward

    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    hist_vol = np.std(np.diff(closing_prices)) * 100  # Historical volatility in percentage

    # Define thresholds
    threshold = 2 * hist_vol if hist_vol > 0 else 0

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    elif position == 1:  # Holding
        if recent_return < -threshold:  # Downtrend
            reward -= 50
        else:  # Uptrend or stable
            reward += 20

    return np.clip(reward, -100, 100)