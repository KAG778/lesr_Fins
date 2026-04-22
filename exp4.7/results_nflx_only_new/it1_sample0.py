import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')[-1]

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema[-1]  # Return the last EMA value

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Calculate indicators
    sma5 = calculate_sma(closing_prices, 5)
    sma10 = calculate_sma(closing_prices, 10)
    ema5 = calculate_ema(closing_prices, 5)
    ema10 = calculate_ema(closing_prices, 10)
    rsi = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # Create enhanced state
    enhanced_s = np.concatenate([
        s,
        np.array([sma5, sma10, ema5, ema10, rsi, atr])
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 1  # Avoid division by zero
    threshold = 2 * historical_vol  # Use relative threshold

    reward = 0
    
    # Determine reward based on recent return and trend indicators
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum

    # Assess trend using SMA or EMA
    sma5 = enhanced_s[-6]  # Last SMA 5
    sma10 = enhanced_s[-5]  # Last SMA 10
    
    if sma5 > sma10:
        reward += 20  # Uptrend
    elif sma5 < sma10:
        reward -= 20  # Downtrend

    # Controlled risk assessment
    if np.abs(recent_return) > historical_vol:  # Using historical volatility as a risk measure
        reward -= 20  # High risk penalty

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]