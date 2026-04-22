import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])  # Start with SMA for the first value
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate the Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, window):
    """Calculate the Average True Range (ATR)."""
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    return np.mean(tr[-window:]) if len(tr) >= window else 0

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    features = []

    # Trend Indicators
    for window in [5, 10, 20]:
        sma = calculate_sma(closing_prices, window)[-1] if len(closing_prices) >= window else np.nan
        features.append(sma)
        ema = calculate_ema(closing_prices, window)[-1] if len(closing_prices) >= window else np.nan
        features.append(ema)

    # Price Relative to Moving Averages
    for window in [5, 10, 20]:
        price_vs_sma = closing_prices[-1] / calculate_sma(closing_prices, window)[-1] - 1 if len(closing_prices) >= window else np.nan
        features.append(price_vs_sma)

    # Momentum Indicators
    for window in [5, 10, 14]:
        rsi = calculate_rsi(closing_prices, window)
        features.append(rsi)

    # Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) if len(closing_prices) > 1 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices) / closing_prices[:-1]) if len(closing_prices) > 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    features.extend([historical_volatility_5, historical_volatility_20, atr])

    # Volume Indicators
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))  # On-Balance Volume
    avg_volume_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    features.extend([obv, avg_volume_5 / (avg_volume_20 + 1e-10) if avg_volume_20 > 0 else 0])  # Volume ratio

    # Market Regime Detection
    volatility_ratio = historical_volatility_5 / (historical_volatility_20 + 1e-10) if historical_volatility_20 > 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2  # R²
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices) + 1e-10) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = avg_volume_5 / (avg_volume_20 + 1e-10) if avg_volume_20 > 0 else 0
    features.extend([volatility_ratio, trend_strength, price_position, volume_ratio_regime])

    # New Features
    features.append(closing_prices[-1] / closing_prices[-2] - 1 if len(closing_prices) > 1 else 0)  # Daily return
    features.append(np.mean(closing_prices[-5:]))  # 5-day average price
    features.append(np.std(closing_prices[-5:]))  # 5-day standard deviation
    features.append(np.max(closing_prices[-5:]) / np.min(closing_prices[-5:]) if np.min(closing_prices[-5:]) != 0 else 0)  # Price range in last 5 days
    features.append(np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else 0)  # Volume ratio
    features.append(np.mean(closing_prices[-10:]) - np.mean(closing_prices[-20:]))  # Difference between 10-day and 20-day average prices
    features.append(np.mean(closing_prices[-3:]) / np.mean(closing_prices[-5:]) - 1)  # Short-term vs Long-term average
    features.append((closing_prices[-1] - np.mean(closing_prices[-5:])) / np.std(closing_prices[-5:]) if np.std(closing_prices[-5:]) > 0 else 0)  # Z-score
    features.append(np.log(closing_prices[-1] / closing_prices[-2]) * 100 if len(closing_prices) > 1 else 0)  # Log return

    # Compile features into enhanced state
    enhanced_s = np.concatenate((s, features))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0

    # Calculate historical volatility
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else 0
    threshold = 2 * historical_vol  # Use volatility-adaptive threshold

    # Initialize reward
    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Buy signal
            reward += 50
        elif recent_return < -threshold:  # Avoiding sharp losses
            reward -= 20
    else:  # Currently holding
        if recent_return > 0:  # Positive trend continuation
            reward += 10
        elif recent_return < -threshold:  # Weakening trend signal
            reward -= 50

    return np.clip(reward, -100, 100)  # Ensure reward is in [-100, 100]