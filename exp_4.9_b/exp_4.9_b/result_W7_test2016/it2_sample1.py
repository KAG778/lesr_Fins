import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
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

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    features = []

    # Multi-timeframe Trend Indicators
    for window in [5, 10, 20]:
        features.append(calculate_sma(closing_prices, window)[-1] if len(closing_prices) >= window else np.nan)
        features.append(calculate_ema(closing_prices, window)[-1] if len(closing_prices) >= window else np.nan)

    # Price vs Moving Averages
    for window in [5, 10]:
        sma = calculate_sma(closing_prices, window)[-1] if len(closing_prices) >= window else np.nan
        if sma is not np.nan and sma != 0:
            features.append(closing_prices[-1] / sma - 1)

    # Momentum Indicators
    for window in [5, 14]:
        features.append(calculate_rsi(closing_prices, window))

    # Volatility Indicators
    for window in [5, 14, 20]:
        historical_volatility = np.std(np.diff(closing_prices[-window:]) / closing_prices[-window-1:-1]) if len(closing_prices) > window else np.nan
        features.append(historical_volatility)

    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    features.append(atr)

    # Volume Indicators
    obv = np.sum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))  # On-Balance Volume
    features.append(obv)
    
    avg_volume_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    features.append(avg_volume_5 / (avg_volume_20 + 1e-10))  # Volume ratio

    # Market Regime Detection
    volatility_ratio = features[-2] / (features[-3] + 1e-10) if features[-3] > 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] ** 2  # R² of closing prices
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices) + 1e-10) if np.max(closing_prices) != np.min(closing_prices) else 0
    
    features.append(volatility_ratio)
    features.append(trend_strength)
    features.append(price_position)

    # New Features
    log_return = np.log(closing_prices[-1] / closing_prices[-2]) * 100 if len(closing_prices) > 1 else 0
    features.append(log_return)

    rolling_mean_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    features.append(rolling_mean_5)

    rolling_std_5 = np.std(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    features.append(rolling_std_5)

    # Combine original state with new features
    enhanced_s = np.concatenate((s, features))

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
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 20  # Adjusted for avoiding sharp losses
    else:  # Currently holding
        if recent_return > 0:  # Positive trend continuation
            reward += 10
        elif recent_return < -threshold:  # Weakening trend signal
            reward -= 50

    return np.clip(reward, -100, 100)  # Ensure reward is in [-100, 100]