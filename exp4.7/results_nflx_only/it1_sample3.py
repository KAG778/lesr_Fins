import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0

    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, window, num_std):
    """Calculate Bollinger Bands."""
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[:len(sma)], ddof=1)
    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)
    return upper_band, lower_band

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volume = s[80:100]
    
    # Calculate additional features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Calculate Bollinger Bands
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20, 2)

    # Handle edge cases for SMA, RSI, and Bollinger Bands calculations
    sma_5 = np.pad(sma_5, (4, 0), constant_values=np.nan)
    sma_10 = np.pad(sma_10, (9, 0), constant_values=np.nan)
    rsi_14 = np.pad(np.array([rsi_14]), (13, 0), constant_values=np.nan)
    upper_band = np.pad(upper_band, (19, 0), constant_values=np.nan)
    lower_band = np.pad(lower_band, (19, 0), constant_values=np.nan)

    # Concatenate all features into enhanced state
    enhanced_s = np.concatenate((s, sma_5, sma_10, ema_10, rsi_14, upper_band, lower_band), axis=0)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    threshold = 2 * historical_vol  # Adaptive threshold
    
    reward = 0

    # Determine reward based on recent return and volatility threshold
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum

    # Adding more reward logic based on RSI and Bollinger Bands
    rsi_value = enhanced_s[-5]  # Last value of the RSI (14)
    upper_band_value = enhanced_s[-1]  # Last value of the upper Bollinger Band
    lower_band_value = enhanced_s[-2]  # Last value of the lower Bollinger Band
    
    if rsi_value < 30 and closing_prices[-1] < lower_band_value:
        reward += 20  # Oversold condition
    elif rsi_value > 70 and closing_prices[-1] > upper_band_value:
        reward -= 20  # Overbought condition

    # Final reward clamping
    reward = np.clip(reward, -100, 100)

    return reward