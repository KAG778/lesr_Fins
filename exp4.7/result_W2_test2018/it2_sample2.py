import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]  # Start EMA with the first data point
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) > window else np.nan
    avg_loss = np.mean(loss[-window:]) if len(loss) > window else np.nan
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.zeros(len(closes))
    atr[window-1] = np.mean(tr[:window])  # Set initial ATR value
    for i in range(window, len(atr)):
        atr[i] = (atr[i-1] * (window - 1) + tr[i - 1]) / window  # SMA of TR
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_12 = calculate_ema(closing_prices, 12)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Prepare enhanced state
    enhanced_s = np.concatenate([
        s,
        sma_5[-1:] if len(sma_5) > 0 else np.array([np.nan]),  # Last value of SMA 5
        sma_10[-1:] if len(sma_10) > 0 else np.array([np.nan]),  # Last value of SMA 10
        ema_12[-1:],  # Last value of EMA 12
        rsi_14,     # Last value of RSI
        atr_14[-1:] if len(atr_14) > 0 else np.array([np.nan])  # Last value of ATR
    ])
    
    # Handle NaN values
    enhanced_s = np.nan_to_num(enhanced_s)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Calculate recent return
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0  # Historical volatility

    # Define adaptive thresholds
    positive_threshold = 2 * historical_vol
    negative_threshold = -2 * historical_vol
    
    # Initialize reward
    reward = 0
    
    # Reward logic based on recent return and RSI
    rsi = enhanced_s[123]  # Assuming the last feature is RSI
    if recent_return > positive_threshold and rsi < 70:
        reward += 50  # Favorable condition for buying
    elif recent_return < negative_threshold and rsi > 30:
        reward -= 50  # Unfavorable condition for selling
    elif 30 < rsi < 70:
        reward += 10  # Neutral but stable condition
    
    return np.clip(reward, -100, 100)  # Clip reward to be within [-100, 100]