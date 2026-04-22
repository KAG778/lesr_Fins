import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    ema = np.zeros(len(prices))
    alpha = 2 / (window + 1)
    ema[window-1] = np.mean(prices[:window])  # Start with SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = calculate_sma(gain, window)
    avg_loss = calculate_sma(loss, window)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate((np.full(window-1, np.nan), rsi))

def revise_state(s):
    closing_prices = s[0:20]
    
    # Adding Simple Moving Average (SMA) for 5, 10, and 20 days
    sma_5 = np.concatenate((np.full(4, np.nan), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.full(9, np.nan), calculate_sma(closing_prices, 10)))
    sma_20 = np.concatenate((np.full(19, np.nan), calculate_sma(closing_prices, 20)))

    # Adding Exponential Moving Average (EMA) for 5, 10, and 20 days
    ema_5 = np.concatenate((np.full(4, np.nan), calculate_ema(closing_prices, 5)))
    ema_10 = np.concatenate((np.full(9, np.nan), calculate_ema(closing_prices, 10)))
    ema_20 = np.concatenate((np.full(19, np.nan), calculate_ema(closing_prices, 20)))

    # Adding Relative Strength Index (RSI) for 14 days
    rsi_14 = np.concatenate((np.full(13, np.nan), calculate_rsi(closing_prices, 14)))

    # Adding Bollinger Bands (20-day)
    sma_20_full = sma_20[19:]  # This is the valid SMA
    std_dev_20 = np.std(closing_prices[-20:])
    upper_band = sma_20_full + (2 * std_dev_20)
    lower_band = sma_20_full - (2 * std_dev_20)

    # Construct the enhanced state
    enhanced_s = np.concatenate((s, 
                                  sma_5, sma_10, sma_20, 
                                  ema_5, ema_10, ema_20, 
                                  rsi_14, 
                                  upper_band, lower_band))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    threshold = 2 * historical_vol  # Adaptive threshold

    reward = 0
    
    # Reward structure based on recent return
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Consider the RSI for risk assessment
    rsi = enhanced_s[100:120][-1]  # Last value of RSI
    if rsi < 30:  # Oversold condition
        reward += 10
    elif rsi > 70:  # Overbought condition
        reward -= 10
    
    # Return intrinsic reward
    return np.clip(reward, -100, 100)