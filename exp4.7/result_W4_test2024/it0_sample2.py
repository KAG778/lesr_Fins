import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * (prices[i] - ema[i-1]) + ema[i-1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Calculate additional features
    sma_5 = np.concatenate((np.array([np.nan]*4), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.array([np.nan]*9), calculate_sma(closing_prices, 10)))
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = np.concatenate((np.array([np.nan]*13), [calculate_rsi(closing_prices, 14)]))
    
    # Volatility calculation (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    
    # Combine all features into enhanced state
    enhanced_s = np.concatenate((s, sma_5, sma_10, ema_10, rsi_14, [historical_vol]))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    historical_vol = enhanced_s[-1]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Define adaptive threshold
    threshold = 2 * historical_vol  # 2x historical volatility
    
    reward = 0
    
    # Reward for positive momentum (recent return)
    if recent_return > threshold:
        reward += 50
    elif recent_return < -threshold:
        reward -= 50

    # Additional checks for trends
    if enhanced_s[20] > enhanced_s[21]:  # Assume SMA_5 > SMA_10 is a positive trend
        reward += 20
    elif enhanced_s[20] < enhanced_s[21]:
        reward -= 20

    # Risk check (daily loss limit)
    if recent_return < -5:  # Check for excessive loss
        reward -= 30
    
    return np.clip(reward, -100, 100)