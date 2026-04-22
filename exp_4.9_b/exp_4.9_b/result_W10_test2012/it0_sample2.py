import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema[-1]

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    if len(prices) < 26:  # MACD requires at least 26 points
        return (np.nan, np.nan, np.nan)
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema(prices, 9)  # Signal line is EMA of MACD
    histogram = macd_line - signal_line
    return (macd_line, signal_line, histogram)

def calculate_atr(highs, lows, closes, window):
    if len(highs) < window:
        return np.nan
    tr = np.maximum(highs[-window:] - lows[-window:], 
                    np.maximum(np.abs(highs[-window:] - closes[-window:]), 
                               np.abs(lows[-window:] - closes[-window:])))
    return np.mean(tr)

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adj_closing_prices = s[100:120]
    
    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    price_vs_sma_5 = closing_prices[-1] - sma_5
    price_vs_sma_10 = closing_prices[-1] - sma_10
    price_vs_sma_20 = closing_prices[-1] - sma_20

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    macd_line, signal_line, histogram = calculate_macd(closing_prices)

    # Volatility Indicators
    historical_vol_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100  # Daily returns
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)
    volatility_ratio = historical_vol_5 / (np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100) if np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) > 0 else np.nan

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))[-1]  # On-Balance Volume
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Market Regime Detection
    volatility_ratio = historical_vol_5 / np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) if np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) > 0 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0][1] ** 2  # R²
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) > np.min(closing_prices[-20:]) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Create the enhanced state
    enhanced_s = np.concatenate([
        s,
        [sma_5, sma_10, sma_20, price_vs_sma_5, price_vs_sma_10, price_vs_sma_20,
         rsi_5, rsi_10, macd_line, signal_line, histogram,
         historical_vol_5, atr_14, volatility_ratio,
         obv, volume_ratio,
         volatility_ratio, trend_strength, price_position, volume_ratio_regime]
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility

    reward = 0
    
    if position_flag == 0:  # Not holding
        if recent_return > threshold and enhanced_s[24] > 70:  # RSI and return condition for buying
            reward += 50  # Strong buy signal
        elif recent_return < -threshold:  # Negative signal
            reward -= 25

    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # Sell signal
            reward -= 50
        elif enhanced_s[24] < 30:  # RSI condition for holding
            reward += 25

    return np.clip(reward, -100, 100)