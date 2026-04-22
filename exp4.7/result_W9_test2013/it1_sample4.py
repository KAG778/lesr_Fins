import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.nan
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else np.nan
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]

    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Create enhanced state, filling with NaN where necessary
    enhanced_s = np.concatenate([
        s,
        np.concatenate((np.full(4, np.nan), sma_5)),
        np.concatenate((np.full(9, np.nan), sma_10)),
        np.concatenate((np.full(4, np.nan), ema_5)),
        np.array([rsi_14]),
        np.array([atr_14])
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return in percentage
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # Historical volatility

    # Define adaptive thresholds
    threshold = 2 * historical_volatility if historical_volatility != 0 else 1  # Prevent division by zero

    reward = 0

    # Reward logic based on recent return and thresholds
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    
    # Include additional checks for risk management based on RSI and volatility
    rsi = enhanced_s[-3]  # Last calculated RSI
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    # Check historical volatility to penalize high risk
    if historical_volatility > 5:  # Example threshold for high volatility
        reward -= 20  # Penalize for high risk

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]