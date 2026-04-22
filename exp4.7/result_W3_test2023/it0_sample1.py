import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[:window])
    avg_loss = np.mean(loss[:window])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adjusted_closing_prices = s[100:119]

    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = np.concatenate((np.full(13, np.nan), [calculate_rsi(closing_prices, 14)]))  # Handle edge case for RSI
    avg_volume = np.mean(volumes)
    
    # Extend the state with new features
    enhanced_s = np.concatenate([s, sma_5[-1:], sma_10[-1:], sma_20[-1:], ema_5[-1:], ema_10[-1:], rsi_14[-1:], [avg_volume]])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100

    # Calculate historical volatility from closing prices
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0

    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol

    reward = 0
    
    # Reward based on return and volatility
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement

    # Add conditions for RSI
    rsi = enhanced_s[-2]  # Assuming RSI is the second last feature
    if rsi < 30:
        reward += 20  # Possible buy opportunity
    elif rsi > 70:
        reward -= 20  # Possible sell opportunity

    # Incorporate risk management
    if recent_return < -5:  # Arbitrary threshold for risk management
        reward -= 30  # Penalty for excessive loss

    # Ensure reward stays within bounds
    reward = np.clip(reward, -100, 100)

    return reward