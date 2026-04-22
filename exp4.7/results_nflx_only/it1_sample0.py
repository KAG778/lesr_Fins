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

def calculate_volatility(prices):
    """Calculate historical volatility."""
    returns = np.diff(prices) / prices[:-1] * 100  # Calculate returns
    return np.std(returns)

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
    volatility = calculate_volatility(closing_prices)

    # Handle edge cases for calculations
    sma_5 = np.pad(sma_5, (4, 0), constant_values=np.nan)
    sma_10 = np.pad(sma_10, (9, 0), constant_values=np.nan)
    rsi_14 = np.pad(np.array([rsi_14]), (13, 0), constant_values=np.nan)

    # Concatenate all features into enhanced state
    enhanced_s = np.concatenate((s, sma_5, sma_10, ema_10, rsi_14, [volatility]), axis=0)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility from closing prices
    historical_vol = enhanced_s[-1]  # Last value is the volatility
    threshold_positive = 2 * historical_vol  # Adaptive threshold for positive returns
    threshold_negative = -2 * historical_vol  # Adaptive threshold for negative returns

    reward = 0

    # Determine reward based on recent return and volatility threshold
    if recent_return > threshold_positive:
        reward += 50  # Positive momentum
    elif recent_return < threshold_negative:
        reward -= 50  # Negative momentum

    # Adding more reward logic based on RSI
    rsi_value = enhanced_s[-5]  # Last value of the RSI (14)
    if rsi_value < 30:
        reward += 20  # Oversold condition
    elif rsi_value > 70:
        reward -= 20  # Overbought condition

    # Final reward clamping
    reward = np.clip(reward, -100, 100)

    return reward