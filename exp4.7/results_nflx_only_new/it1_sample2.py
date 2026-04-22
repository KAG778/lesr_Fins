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
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    """Calculate Average True Range."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    
    # Calculate indicators
    sma5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    ema5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    ema10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    rsi = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14) if len(high_prices) >= 14 and len(low_prices) >= 14 else np.nan

    # Create enhanced state
    enhanced_s = np.concatenate([s,
                                  np.array([sma5, sma10, ema5, ema10, rsi, atr])
                                 ])
    
    # Fill NaN values with zeros for consistent state representation
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
    if recent_return > threshold:
        reward += 50  # Reward for significant positive return
    elif recent_return < -threshold:
        reward -= 50  # High penalty for large negative return

    # Trend assessment using SMA
    sma5 = enhanced_s[-6]  # Last SMA 5
    sma10 = enhanced_s[-5]  # Last SMA 10
    
    if sma5 > sma10:
        reward += 20  # Uptrend confirmation
    elif sma5 < sma10:
        reward -= 20  # Downtrend confirmation
    
    # Risk control
    if np.abs(recent_return) > historical_vol:  # Use historical volatility as a risk threshold
        reward -= 20  # Penalty for high daily return risk

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]