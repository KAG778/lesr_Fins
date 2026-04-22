import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average."""
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Start with the first price
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    adj_closing_prices = s[100:119]
    
    # Calculate new features
    sma_5 = calculate_sma(closing_prices, 5) if len(closing_prices) >= 5 else np.array([np.nan]*16)
    sma_10 = calculate_sma(closing_prices, 10) if len(closing_prices) >= 10 else np.array([np.nan]*11)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    rsi_14 = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan
    
    # Create enhanced state
    enhanced_s = np.concatenate([
        s,
        sma_5,
        sma_10,
        ema_5,
        ema_10,
        np.array([rsi_14] + [np.nan] * (19 - len([rsi_14]))),  # Fill the rest with NaN
    ])
    
    # Handle NaN values by filling with zeros or another method as per your need
    enhanced_s = np.nan_to_num(enhanced_s)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    
    # Calculate daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    recent_return = returns[-1] if len(returns) > 0 else 0
    
    # Calculate historical volatility
    historical_vol = np.std(returns) if len(returns) > 0 else 1  # Avoid division by zero
    threshold = 2 * historical_vol  # Adaptive threshold
    
    reward = 0
    
    # Reward logic based on recent return and volatility
    if recent_return < -threshold:
        reward -= 50  # High penalty for large negative return
    elif recent_return > threshold:
        reward += 50  # Reward for significant positive return
    
    # Additional conditions for risk management
    if np.abs(recent_return) > 5:  # Arbitrary risk level
        reward -= 20  # Penalty for high daily return risk
    
    # Normalize reward to fit within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward