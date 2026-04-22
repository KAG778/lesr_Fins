import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[:window] = np.nan  # First 'window' values will be NaN
    ema[window - 1] = np.mean(prices[:window])  # Setting the first EMA value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], highs[1:] - closes[:-1])
    tr = np.maximum(tr, closes[:-1] - lows[1:])
    atr = np.convolve(tr, np.ones(window) / window, mode='valid')
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    features = []

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else np.nan
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan

    features.extend([sma_5, sma_10, sma_20, ema_5, ema_10])

    # Price position relative to moving averages
    price_position_sma_10 = closing_prices[-1] / sma_10 - 1 if not np.isnan(sma_10) else np.nan
    features.append(price_position_sma_10)

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    rsi_10 = calculate_rsi(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    features.extend([rsi_5, rsi_10])

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)[-1] if len(closing_prices) >= 14 else np.nan
    features.extend([historical_volatility_5, historical_volatility_20, atr])

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:])) if len(closing_prices) > 1 else 0
    avg_volume_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    volume_ratio = avg_volume_5 / (avg_volume_20 + 1e-10)  # Avoid division by zero
    features.extend([obv[-1], volume_ratio])

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / (historical_volatility_20 + 1e-10) if historical_volatility_20 > 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2  # R² of closing prices
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices) + 1e-10) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = avg_volume_5 / (avg_volume_20 + 1e-10) if avg_volume_20 > 0 else 0
    features.extend([volatility_ratio, trend_strength, price_position, volume_ratio_regime])

    # Compile features into enhanced state
    enhanced_s = np.concatenate((s, features))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility and use it for adaptive thresholds
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else 0
    threshold = 2 * historical_vol  # Use 2x standard deviation for the threshold

    # Initialize reward
    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 20  # Reduced penalty for strong downtrend to avoid excessive loss
    else:  # Currently holding
        if recent_return > 0:  # Positive trend continuation
            reward += 10
        elif recent_return < -threshold:  # Trend weakening
            reward -= 50  # Strong penalty for weakening trend

    return np.clip(reward, -100, 100)  # Ensure reward is in [-100, 100]