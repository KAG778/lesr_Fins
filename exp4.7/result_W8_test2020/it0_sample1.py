import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.mean(gain)
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else np.mean(loss)

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.convolve(tr, np.ones(window)/window, mode='valid')
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    trading_volume = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Create enhanced state
    enhanced_s = np.concatenate((
        s, 
        sma_5, sma_10, sma_20,
        ema_5, ema_10,
        np.array([rsi_14]), 
        atr_14
    ))

    # Handle edge case for length issues (pad with NaNs or zeros)
    while len(enhanced_s) < 120:
        enhanced_s = np.append(enhanced_s, [np.nan])
    
    # Ensure the enhanced state is 120-dimensional
    return enhanced_s[:120]

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol
    
    reward = 0

    # Check recent return against the threshold
    if recent_return < -threshold:
        reward -= 50
    elif recent_return > threshold:
        reward += 50

    # Use RSI as a feature for trend identification
    rsi = enhanced_s[119]  # Assuming the last index is RSI
    if rsi < 30:
        reward += 20  # Oversold
    elif rsi > 70:
        reward -= 20  # Overbought

    return np.clip(reward, -100, 100)