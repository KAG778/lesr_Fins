import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * alpha + ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, window):
    tr = np.maximum(high[1:] - low[1:], 
                   np.maximum(np.abs(high[1:] - close[:-1]), 
                              np.abs(low[1:] - close[:-1])))
    atr = np.zeros(len(close))
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i - 1]) / window
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]

    # Calculate features
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    atr_14 = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Create enhanced state with the latest values
    enhanced_s = np.concatenate([
        s,
        np.pad(sma_5[-1:], (0, 15), 'constant', constant_values=np.nan),
        np.pad(sma_10[-1:], (0, 10), 'constant', constant_values=np.nan),
        np.pad(ema_5[-1:], (0, 15), 'constant', constant_values=np.nan),
        np.pad(ema_10[-1:], (0, 10), 'constant', constant_values=np.nan),
        np.array([rsi_14]), 
        atr_14[-1:]
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_volatility = np.std(returns)  # Historical volatility

    if len(returns) == 0:
        return 0  # Avoid division by zero

    recent_return = returns[-1] if len(returns) > 0 else 0
    threshold = 2 * historical_volatility  # Adaptive threshold based on historical volatility

    reward = 0

    # Reward based on recent return and adaptive thresholds
    if recent_return > threshold:
        reward += 50  # Strong positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Strong negative momentum

    # Include additional checks for risk management
    rsi = enhanced_s[-3]  # Last calculated RSI
    if rsi < 30:
        reward += 20  # Oversold condition, positive adjustment
    elif rsi > 70:
        reward -= 20  # Overbought condition, negative adjustment

    # Additional volatility penalties
    if historical_volatility > 5:  # High risk observed, penalize
        reward -= 20

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]