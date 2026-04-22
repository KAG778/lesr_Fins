import numpy as np

def compute_sma(prices, window):
    """Compute Simple Moving Average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def compute_ema(prices, window):
    """Compute Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[window - 1] = np.mean(prices[:window])  # start with SMA for the first value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def compute_rsi(prices, window):
    """Compute Relative Strength Index."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:]) if len(gains) >= window else 0
    avg_loss = np.mean(losses[-window:]) if len(losses) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(highs, lows, closes, window):
    """Compute Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(window) / window, mode='valid')
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]

    # Multi-timeframe Trend Indicators
    sma_5 = compute_sma(closing_prices, 5)
    sma_10 = compute_sma(closing_prices, 10)
    sma_20 = compute_sma(closing_prices, 20)
    ema_5 = compute_ema(closing_prices, 5)
    ema_10 = compute_ema(closing_prices, 10)
    
    price_vs_sma_5 = closing_prices[-1] / sma_5[-1] if len(sma_5) > 0 else 0
    price_vs_sma_10 = closing_prices[-1] / sma_10[-1] if len(sma_10) > 0 else 0
    
    trend_diff_5_10 = ema_5[-1] - ema_10[-1] if len(ema_5) > 0 and len(ema_10) > 0 else 0

    # Momentum Indicators
    rsi_5 = compute_rsi(closing_prices, 5)
    rsi_10 = compute_rsi(closing_prices, 10)
    rsi_14 = compute_rsi(closing_prices, 14)

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) if len(closing_prices) > 1 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) if len(closing_prices) > 20 else 0
    atr = compute_atr(high_prices, low_prices, closing_prices, 14)[-1] if len(closing_prices) > 14 else 0

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volume[1:], np.where(np.diff(closing_prices) < 0, -volume[1:], 0)))
    volume_ratio = np.mean(volume[-5:]) / np.mean(volume) if np.mean(volume) != 0 else 0

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 != 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2  # R²
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volume[-5:]) / np.mean(volume[-20:]) if np.mean(volume[-20:]) != 0 else 0

    # Create enhanced state
    enhanced_s = np.concatenate((
        s,  # Original state
        [sma_5[-1] if len(sma_5) > 0 else 0,
         sma_10[-1] if len(sma_10) > 0 else 0,
         sma_20[-1] if len(sma_20) > 0 else 0,
         ema_5[-1] if len(ema_5) > 0 else 0,
         ema_10[-1] if len(ema_10) > 0 else 0,
         price_vs_sma_5,
         price_vs_sma_10,
         trend_diff_5_10,
         rsi_5,
         rsi_10,
         rsi_14,
         historical_volatility_5,
         historical_volatility_20,
         atr,
         obv[-1] if len(obv) > 0 else 0,
         volume_ratio,
         volatility_ratio,
         trend_strength,
         price_position,
         volume_ratio_regime]
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility

    reward = 0

    if position == 0:  # Not holding
        if recent_return > threshold:
            reward += 50  # Strong buy signal
        else:
            reward -= 10  # Penalize uncertain conditions
    else:  # Holding
        if recent_return < -threshold:
            reward -= 50  # Strong sell signal
        elif recent_return > 0:
            reward += 10  # Reward for holding during positive returns
        else:
            reward -= 10  # Penalize if returns are neutral or negative

    return np.clip(reward, -100, 100)  # Clip reward to the range of [-100, 100]