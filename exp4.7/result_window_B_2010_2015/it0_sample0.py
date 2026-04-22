import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average (SMA)"""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average (EMA)"""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index (RSI)"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:120]

    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    rsi_14 = np.array([calculate_rsi(closing_prices[:i+1], 14) if i >= 13 else np.nan for i in range(len(closing_prices))])[-1]

    # Prepare enhanced state
    enhanced_s = np.concatenate((
        s,
        sma_5[-1:],  # Last value of SMA 5
        sma_10[-1:],  # Last value of SMA 10
        sma_20[-1:],  # Last value of SMA 20
        ema_5[-1:],  # Last value of EMA 5
        ema_10[-1:],  # Last value of EMA 10
        np.array([rsi_14])  # Last value of RSI
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    rsi = enhanced_s[120]  # The last item in the enhanced state is RSI

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 1e-6  # Prevent division by zero
    threshold = 2 * historical_vol  # Set threshold at 2x historical volatility

    reward = 0

    # Reward based on recent return and RSI
    if recent_return > threshold:
        reward += 50  # Positive reward for good positive return
    elif recent_return < -threshold:
        reward -= 50  # Negative reward for bad return

    # Additional reward/penalty based on RSI
    if rsi > 70:
        reward -= 20  # Overbought condition
    elif rsi < 30:
        reward += 20  # Oversold condition

    return np.clip(reward, -100, 100)  # Ensure the reward is within [-100, 100]