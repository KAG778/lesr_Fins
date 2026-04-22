import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
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
    return np.mean(tr[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closes = s[100:120]

    # Calculate indicators
    sma5 = calculate_sma(closing_prices, 5)
    sma10 = calculate_sma(closing_prices, 10)
    sma20 = calculate_sma(closing_prices, 20)
    
    ema5 = calculate_ema(closing_prices, 5)
    ema10 = calculate_ema(closing_prices, 10)
    
    rsi = calculate_rsi(closing_prices, 14)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Create enhanced state
    enhanced_s = np.concatenate([s, sma5[-1:], sma10[-1:], sma20[-1:], 
                                  ema5[-1:], ema10[-1:], 
                                  np.array([rsi]), np.array([atr])])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)

    # Use relative threshold (2x historical volatility)
    threshold = 2 * historical_vol
    
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

    # Reward for controlled risk
    if np.abs(recent_return) > 5:  # 5% is arbitrary, can be adjusted
        reward -= 20  # High risk

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]