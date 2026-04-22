import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window), 'valid') / window

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Starting point
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

def calculate_volatility(prices):
    returns = np.diff(prices) / prices[:-1] * 100
    return np.std(returns)

def revise_state(s):
    closing_prices = s[0:20]
    volumes = s[80:100]  # Volume data
    
    # Calculate moving averages
    sma_5 = np.concatenate((np.full(4, np.nan), calculate_sma(closing_prices, 5)))
    sma_10 = np.concatenate((np.full(9, np.nan), calculate_sma(closing_prices, 10)))
    sma_20 = np.concatenate((np.full(19, np.nan), calculate_sma(closing_prices, 20)))
    
    ema_5 = np.concatenate((np.full(4, np.nan), calculate_ema(closing_prices, 5)))
    ema_10 = np.concatenate((np.full(9, np.nan), calculate_ema(closing_prices, 10)))
    
    rsi_14 = np.concatenate((np.full(13, np.nan), np.array([calculate_rsi(closing_prices, 14)])))
    
    # Calculate historical volatility
    historical_volatility = calculate_volatility(closing_prices)
    
    # Construct enhanced state
    enhanced_s = np.concatenate((s, 
                                  sma_5, sma_10, sma_20, 
                                  ema_5, ema_10, 
                                  rsi_14, 
                                  np.array([historical_volatility])))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0
    historical_volatility = enhanced_s[-1]  # Assuming last element is historical volatility
    
    # Use relative thresholds based on historical volatility
    threshold = 2 * historical_volatility  # Volatility-adaptive threshold
    reward = 0

    # Reward logic
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
    
    # Add conditions based on RSI
    rsi = enhanced_s[139]  # Assuming it's the 140th feature
    if rsi < 30:
        reward += 20  # Oversold condition
    elif rsi > 70:
        reward -= 20  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]