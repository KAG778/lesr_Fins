import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate((np.nan * np.ones(window-1), rsi))

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[11:] - ema_26[25:]  # Align lengths
    return np.concatenate((np.nan * np.ones(25), macd))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]
    adjusted_close = s[100:119]

    enhanced_s = np.zeros(120 + 6)  # Original 120 + 6 new features

    # Copy original state
    enhanced_s[0:120] = s

    # Calculate new features
    enhanced_s[120] = np.mean(closing_prices)  # Mean closing price
    enhanced_s[121] = np.std(closing_prices)   # Std. dev of closing prices
    enhanced_s[122] = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan  # 5-day SMA
    enhanced_s[123] = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan  # 10-day SMA
    enhanced_s[124] = calculate_rsi(closing_prices, 14)[-1] if len(closing_prices) >= 14 else np.nan  # 14-day RSI
    enhanced_s[125] = calculate_macd(closing_prices)[-1] if len(closing_prices) >= 26 else np.nan  # MACD

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    historical_volatility = enhanced_s[121]
    
    # Calculate recent return
    recent_return = ((closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]) * 100 if closing_prices[-2] != 0 else 0
    
    # Volatility-adaptive thresholds
    threshold = 2 * historical_volatility  # 2x historical volatility

    reward = 0

    # Determine reward based on recent return and volatility
    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Adjust reward based on the RSI
    rsi = enhanced_s[124]
    if rsi < 30:
        reward += 20  # Oversold condition, potential buy signal
    elif rsi > 70:
        reward -= 20  # Overbought condition, potential sell signal

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]