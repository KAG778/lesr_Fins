import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[window - 1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema[-1]

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_volatility(returns):
    """Calculate historical volatility."""
    return np.std(returns) * 100  # Convert to percentage

def calculate_atr(highs, lows, closes):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr)

def revise_state(s):
    # Extracting raw OHLCV data
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    enhanced_features = []

    # Trend Indicators
    for window in [5, 10, 20, 50]:
        sma = calculate_sma(closing_prices, window)
        ema = calculate_ema(closing_prices, window)
        enhanced_features.append(sma)  # SMA
        enhanced_features.append(ema)  # EMA
        enhanced_features.append(closing_prices[-1] / sma if sma != 0 else np.nan)  # Price/SMA ratio
        enhanced_features.append(closing_prices[-1] / ema if ema != 0 else np.nan)  # Price/EMA ratio

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    enhanced_features.extend([rsi_5, rsi_14])

    # Volatility Indicators
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = calculate_volatility(daily_returns)
    atr = calculate_atr(high_prices, low_prices, closing_prices)
    enhanced_features.extend([historical_volatility, atr])

    # Volume Indicators
    obv = np.cumsum(np.where(np.diff(adjusted_closing_prices) > 0, volumes[1:], 
                              np.where(np.diff(adjusted_closing_prices) < 0, -volumes[1:], 0)))
    volume_avg_5 = calculate_sma(volumes, 5)
    volume_avg_20 = calculate_sma(volumes, 20)
    enhanced_features.extend([obv[-1], volume_avg_5, volume_avg_20])
    
    # Market Regime Detection
    volatility_ratio = historical_volatility / np.mean(daily_returns) if np.mean(daily_returns) != 0 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # Linear regression R²
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = volume_avg_5 / volume_avg_20 if volume_avg_20 != 0 else 0

    enhanced_features.extend([volatility_ratio, trend_strength, price_position, volume_ratio_regime])

    # Combine original state with enhanced features
    enhanced_s = np.concatenate((s, np.array(enhanced_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Position flag (1.0=holding, 0.0=not holding)
    closing_prices = enhanced_s[0:20]
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns

    if len(daily_returns) < 1:
        return 0  # Insufficient data for reward calculation

    recent_return = daily_returns[-1] * 100  # Convert to percentage
    historical_vol = calculate_volatility(daily_returns)  # Convert to percentage
    threshold = 2 * historical_vol if historical_vol != 0 else 0  # Volatility-adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if enhanced_s[-5] < 30 and recent_return > threshold:  # Assuming RSI < 30 indicates oversold
            reward += 50  # Strong buy signal
        elif recent_return < -threshold:
            reward -= 30  # Negative return penalization
        else:
            reward += 10  # Neutral reward

    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # If recent return is significantly negative
            reward -= 50  # Penalizing strong down moves
        elif enhanced_s[-5] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 20  # Positive reward for taking profit
        else:
            reward += 10  # Neutral reward

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]