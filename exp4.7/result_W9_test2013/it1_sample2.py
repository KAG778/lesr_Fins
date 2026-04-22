import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    alpha = 2 / (window + 1)
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])  # Use mean for ATR over the window

def calculate_volume_change(volumes):
    return np.diff(volumes) / volumes[:-1] * 100

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)[-1:]  # Last SMA
    sma_10 = calculate_sma(closing_prices, 10)[-1:]  # Last SMA
    ema_5 = calculate_ema(closing_prices, 5)[-1:]  # Last EMA
    rsi_14 = calculate_rsi(closing_prices, 14)  # Last RSI
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)  # Last ATR
    volume_change = calculate_volume_change(volumes)[-1:]  # Last volume change percentage

    # Create enhanced state
    enhanced_s = np.concatenate((s, sma_5, sma_10, ema_5, np.array([rsi_14]), np.array([atr_14]), volume_change))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    
    # Calculate historical volatility
    historical_vol = np.std(returns)
    
    # Calculate recent return
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Define adaptive thresholds
    threshold = 2 * historical_vol  # Use 2x historical volatility as threshold
    
    # Initialize reward
    reward = 0

    # Reward calculation based on recent return and thresholds
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Additional checks for risk management
    if historical_vol > 5:  # High risk observed, penalize
        reward -= 20  # Penalize for high volatility

    # RSI Adjustment
    rsi = enhanced_s[-3]  # Last calculated RSI
    if rsi < 30:
        reward += 20  # Oversold condition, incentivize buying
    elif rsi > 70:
        reward -= 20  # Overbought condition, incentivize selling

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]