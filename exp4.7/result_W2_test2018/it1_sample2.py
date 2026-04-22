import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    ema = np.zeros(len(prices))
    alpha = 2 / (window + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, window):
    """Calculate Average True Range (ATR)"""
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros(len(close))
    atr[window - 1] = np.mean(tr[:window])
    for i in range(window, len(atr)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    rsi = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Prepare enhanced state
    enhanced_s = np.concatenate([
        s,
        sma_5[-1:],  # Last value of SMA 5
        sma_10[-1:], # Last value of SMA 10
        ema_5[-1:],  # Last value of EMA 5
        rsi,         # Current value of RSI
        atr[-1:]     # Last value of ATR
    ])

    # Handle edge cases (e.g., NaN values for indicators)
    enhanced_s = np.nan_to_num(enhanced_s, nan=0.0)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Calculate recent return
    
    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)  # Historical volatility

    # Define thresholds based on historical volatility
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold
    
    # Initialize reward
    reward = 0
    
    # Reward logic based on recent return and RSI
    rsi = enhanced_s[123]  # The last value is assumed to be RSI
    if recent_return > threshold and rsi < 70:
        reward += 50  # Favorable condition for buying
    elif recent_return < -threshold and rsi > 30:
        reward -= 50  # Unfavorable condition for selling
    elif 30 < rsi < 70:
        reward += 10  # Neutral but stable condition

    # Add additional reward for ATR indicating low volatility (risk control)
    if enhanced_s[124] < (historical_vol * 0.5):  # If ATR is less than half of historical volatility
        reward += 10  # Reward for low volatility

    return np.clip(reward, -100, 100)  # Clip reward to be within [-100, 100]