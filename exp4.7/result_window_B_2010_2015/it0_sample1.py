import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Start EMA with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    if avg_loss == 0:
        return 100  # Avoid division by zero; max RSI
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculate additional features
    enhanced_s = np.copy(s)

    # Calculate 5-day and 10-day SMA
    sma_5 = calculate_sma(closing_prices, 5) if len(closing_prices) >= 5 else np.array([np.nan])
    sma_10 = calculate_sma(closing_prices, 10) if len(closing_prices) >= 10 else np.array([np.nan])

    # Calculate 5-day and 10-day EMA
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan

    # Calculate 14-day RSI
    rsi = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan

    # Add new features to enhanced state
    enhanced_s = np.concatenate((enhanced_s, np.array([sma_5[-1], sma_10[-1], ema_5, ema_10, rsi])))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Percentage return
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    # Define thresholds based on historical volatility
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    # Initialize reward
    reward = 0

    # Assess reward based on recent return and historical volatility
    if recent_return > threshold:
        reward += 50  # Strong positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong negative momentum
    else:
        reward += 10  # Neutral to positive state

    # Incorporate RSI into the reward evaluation
    rsi = enhanced_s[-1]  # Last feature added in revised state
    if rsi < 30:
        reward -= 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range