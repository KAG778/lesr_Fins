import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[25:] - ema_26[25:]  # Align lengths
    return macd

def calculate_bollinger_bands(prices, window):
    sma = calculate_sma(prices, window)
    rolling_std = np.std(prices[-window:])
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    return upper_band, lower_band

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volume = s[80:99]
    adjusted_closing_prices = s[100:119]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)
    upper_band, lower_band = calculate_bollinger_bands(closing_prices, 20)
    
    # Handle edge cases for new features
    extended_length = max(len(sma_5), len(sma_10), len(sma_20), len(ema_10), len(macd))
    
    # Create enhanced state
    enhanced_s = np.concatenate((
        s, 
        np.pad(sma_5, (extended_length - len(sma_5), 0), 'constant', constant_values=np.nan),
        np.pad(sma_10, (extended_length - len(sma_10), 0), 'constant', constant_values=np.nan),
        np.pad(sma_20, (extended_length - len(sma_20), 0), 'constant', constant_values=np.nan),
        np.pad(ema_10, (extended_length - len(ema_10), 0), 'constant', constant_values=np.nan),
        np.array([rsi_14]),  # Add RSI as a single value
        np.pad(macd, (extended_length - len(macd), 0), 'constant', constant_values=np.nan),
        np.pad(upper_band, (extended_length - len(upper_band), 0), 'constant', constant_values=np.nan),
        np.pad(lower_band, (extended_length - len(lower_band), 0), 'constant', constant_values=np.nan)
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100  # Daily returns
    recent_return = returns[-1]  # Most recent return
    
    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Volatility-adaptive threshold

    reward = 0

    # Evaluate reward based on recent return
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Trend checking using SMA
    sma_5 = np.mean(closing_prices[-5:])
    sma_10 = np.mean(closing_prices[-10:])
    
    if sma_5 > sma_10:
        reward += 20  # Positive trend
    elif sma_5 < sma_10:
        reward -= 20  # Negative trend

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]