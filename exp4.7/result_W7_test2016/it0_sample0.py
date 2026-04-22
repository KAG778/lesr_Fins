import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[window-1] = np.mean(prices[:window])  # Seed with SMA
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[:window])
    avg_loss = np.mean(loss[:window])
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return np.concatenate((np.full(window - 1, np.nan), np.repeat(rsi, len(prices) - window + 1)))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adj_closing_prices = s[100:120]

    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    rsi_14 = calculate_rsi(closing_prices, 14)

    # Pad the new features to match dimensions
    sma_5 = np.concatenate((np.full(4, np.nan), sma_5))
    sma_10 = np.concatenate((np.full(9, np.nan), sma_10))
    sma_20 = np.concatenate((np.full(19, np.nan), sma_20))
    
    ema_5 = np.concatenate((np.full(4, np.nan), ema_5))
    ema_10 = np.concatenate((np.full(9, np.nan), ema_10))
    
    rsi_14 = np.concatenate((np.full(13, np.nan), rsi_14))

    # Combine original state with new features
    enhanced_s = np.concatenate([
        closing_prices, opening_prices, high_prices, low_prices, volumes, adj_closing_prices,
        sma_5, sma_10, sma_20, ema_5, ema_10, rsi_14
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    rsi_14 = enhanced_s[120:133]  # RSI is added after the original dimensions

    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # percentage

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    
    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol

    reward = 0

    # Assess the reward based on recent return and RSI
    if recent_return > threshold:
        reward += 50  # Favorable upward movement
    elif recent_return < -threshold:
        reward -= 50  # Unfavorable downward movement

    # Include RSI in the reward calculation
    if rsi_14[-1] < 30:  # Oversold condition
        reward += 20
    elif rsi_14[-1] > 70:  # Overbought condition
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within the range