import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[:window] = np.nan  # First 'window' values will be NaN
    ema[window-1] = np.mean(prices[:window])  # Setting the first EMA value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closes[:-1])
    tr = np.maximum(tr, closes[:-1] - lows[1:])
    atr = np.convolve(tr, np.ones(window)/window, mode='valid')
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else np.nan
    
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan

    price_vs_sma_5 = closing_prices[-1] - sma_5 if not np.isnan(sma_5) else np.nan
    price_vs_sma_10 = closing_prices[-1] - sma_10 if not np.isnan(sma_10) else np.nan

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    rsi_10 = calculate_rsi(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    rsi_14 = calculate_rsi(closing_prices, 14)[-1] if len(closing_prices) >= 14 else np.nan

    # Volatility Indicators
    historical_vol_5 = np.std(np.diff(closing_prices[:5]) / closing_prices[:4] * 100) if len(closing_prices) >= 5 else np.nan
    historical_vol_20 = np.std(np.diff(closing_prices[:20]) / closing_prices[:19] * 100) if len(closing_prices) >= 20 else np.nan
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)[-1] if len(closing_prices) >= 14 else np.nan

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:])) if len(closing_prices) > 1 else 0
    volume_ratio = volumes[-1] / (np.mean(volumes) + 1e-10)  # Avoid division by zero

    # Market Regime Detection
    volatility_ratio = historical_vol_5 / (historical_vol_20 + 1e-10) if historical_vol_20 > 0 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # R² of closing prices
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices) + 1e-10)
    volume_ratio_regime = np.mean(volumes[-5:]) / (np.mean(volumes[-20:]) + 1e-10) if len(volumes) >= 20 else np.nan

    # Compile features into enhanced state
    enhanced_s = np.concatenate((
        s,
        [sma_5, sma_10, sma_20, ema_5, ema_10, price_vs_sma_5, price_vs_sma_10,
         rsi_5, rsi_10, rsi_14, historical_vol_5, historical_vol_20, atr, obv[-1], volume_ratio,
         volatility_ratio, trend_strength, price_position, volume_ratio_regime]
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility
    threshold = 2 * historical_vol  # Relative threshold based on volatility
    
    reward = 0
    
    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Trend weakening
            reward -= 50
        elif recent_return > threshold:  # Trend strengthening
            reward += 50

    return np.clip(reward, -100, 100)  # Ensure reward is in [-100, 100]