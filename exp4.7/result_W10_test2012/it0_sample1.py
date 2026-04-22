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
    avg_gain = np.mean(gain[-window:])  # use the last window for the average
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd = ema_12[len(ema_12) - len(ema_26):] - ema_26
    return macd

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volume = s[80:99]
    adjusted_closing_prices = s[100:119]

    # Calculate technical indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    rsi = calculate_rsi(closing_prices, 14)
    macd = calculate_macd(closing_prices)

    # Adding features with appropriate padding for alignment
    padded_sma_5 = np.pad(sma_5, (4, 0), 'edge')  # pad the array to match 20 days
    padded_sma_10 = np.pad(sma_10, (9, 0), 'edge')
    padded_rsi = np.pad(np.array([rsi]), (13, 0), 'edge')  # RSI is a single value
    padded_macd = np.pad(macd, (25, 0), 'edge')  # pad MACD to align

    # Combine all features into the enhanced state
    enhanced_s = np.concatenate([s, padded_sma_5, padded_sma_10, padded_rsi, padded_macd])
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)
    
    # Use relative thresholds for reward assignment
    threshold = 2 * historical_vol  # adaptive threshold based on historical volatility

    reward = 0

    if recent_return > threshold:
        reward += 50  # Strong positive return
    elif recent_return < -threshold:
        reward -= 50  # Strong negative return

    # Additional conditions based on RSI or other indicators for more fine-tuned rewards
    rsi = enhanced_s[39]  # Assuming RSI is at position 39 in the enhanced state
    if rsi < 30:
        reward += 10  # Oversold condition
    elif rsi > 70:
        reward -= 10  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within the range [-100, 100]