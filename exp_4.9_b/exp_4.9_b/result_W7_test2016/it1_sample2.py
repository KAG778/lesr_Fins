import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window), 'valid') / window

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])  # Start with SMA for the first value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate the Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, window):
    """Calculate the Average True Range (ATR)."""
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    return np.mean(tr[-window:]) if len(tr) >= window else 0

def calculate_bollinger_bands(prices, window, num_std_dev):
    """Calculate Bollinger Bands."""
    sma = calculate_sma(prices, window)
    std_dev = np.std(prices[-window:])
    upper_band = sma + (num_std_dev * std_dev)
    lower_band = sma - (num_std_dev * std_dev)
    return upper_band, lower_band

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[20:40]
    low_prices = s[40:60]
    volumes = s[60:80]

    features = []

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else np.nan

    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5) if len(closing_prices) >= 5 else np.nan
    rsi_14 = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan

    # Volatility Indicators
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14) if len(closing_prices) >= 14 else np.nan
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:]) / closing_prices[-6:-1]) if len(closing_prices) > 5 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) if len(closing_prices) > 20 else 0

    # Bollinger Bands
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20, 2)

    # Volume Indicators
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))[-1] if len(closing_prices) > 1 else 0
    volume_ratio = volumes[-1] / (np.mean(volumes[-20:]) + 1e-10) if len(volumes) >= 20 else 0

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / (historical_volatility_20 + 1e-10) if historical_volatility_20 > 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2 if len(closing_prices) > 1 else 0
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices) + 1e-10) if np.max(closing_prices) != np.min(closing_prices) else 0

    # Combine features into enhanced state
    enhanced_s = np.concatenate((
        s,  # Original state
        [sma_5, sma_10, sma_20, ema_5, ema_10, rsi_5, rsi_14, historical_volatility_5, historical_volatility_20, atr,
        obv, volume_ratio, upper_band, lower_band, volatility_ratio, trend_strength, price_position]
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else 0
    threshold = 2 * historical_vol  # Use volatility-adaptive threshold

    # Initialize reward
    reward = 0

    if position_flag == 0:  # Not holding
        # Positive reward for strong uptrend
        if recent_return > threshold:
            reward += 50  # Buy signal
        elif recent_return < -threshold:
            reward -= 20  # Avoiding sharp losses
    else:  # Currently holding
        if recent_return > 0:  # Positive trend continuation
            reward += 10
        elif recent_return < -threshold:  # Weakening trend signal
            reward -= 50

    return np.clip(reward, -100, 100)