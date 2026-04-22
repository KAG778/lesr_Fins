import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[:window] = np.nan  # Initial EMA values are NaN
    ema[window-1] = np.mean(prices[:window])  # First EMA is SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
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

def calculate_bollinger_bands(prices, window, num_std_dev):
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[:window])
    upper_band = sma + (rolling_std * num_std_dev)
    lower_band = sma - (rolling_std * num_std_dev)
    return upper_band, lower_band

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculate additional features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)

    ema_5 = calculate_ema(closing_prices, 5)
    rsi_14 = calculate_rsi(closing_prices, 14)
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20, 2)

    # Combine all features into enhanced state
    enhanced_s = np.concatenate((s, 
                                  sma_5, sma_10, sma_20, 
                                  ema_5, rsi_14, 
                                  upper_band, lower_band))
    
    # Handle edge cases: fill NaN values with zeros or the last valid observation
    enhanced_s = np.nan_to_num(enhanced_s)
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    rsi = enhanced_s[100]  # Assuming RSI is the 100th index in enhanced state

    # Calculate daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    recent_return = returns[-1] if len(returns) > 0 else 0

    # Calculate historical volatility
    historical_vol = np.std(returns)

    # Define thresholds relative to historical volatility
    threshold = 2 * historical_vol

    reward = 0
    
    # Reward logic
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return
    else:
        reward += 10  # Mild return, maintain position

    # Reward based on RSI levels
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    # Limit reward to the range of [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward