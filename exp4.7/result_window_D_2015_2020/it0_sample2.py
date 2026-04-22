import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-window:])
    avg_loss = np.mean(loss[-window:])
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    trading_volume = s[80:99]
    adjusted_closing_prices = s[100:119]

    # Enhanced features
    enhanced_s = np.copy(s)

    # Simple Moving Averages (SMA)
    enhanced_s = np.concatenate((enhanced_s, 
                                  calculate_sma(closing_prices, 5)[-1:],  # 5-day SMA
                                  calculate_sma(closing_prices, 10)[-1:], # 10-day SMA
                                  calculate_sma(closing_prices, 20)[-1:])) # 20-day SMA

    # Exponential Moving Averages (EMA)
    enhanced_s = np.concatenate((enhanced_s, 
                                  calculate_ema(closing_prices, 5)[-1:],  # 5-day EMA
                                  calculate_ema(closing_prices, 10)[-1:], # 10-day EMA
                                  calculate_ema(closing_prices, 20)[-1:])) # 20-day EMA

    # Relative Strength Index (RSI)
    rsi = calculate_rsi(closing_prices, 14)  # 14-day RSI
    enhanced_s = np.concatenate((enhanced_s, [rsi]))

    # Calculate recent volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if returns.size > 0 else 0
    enhanced_s = np.concatenate((enhanced_s, [historical_vol]))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_volatility = enhanced_s[-1]  # Last feature in enhanced state is historical volatility

    # Volatility-adaptive thresholds
    threshold = 2 * historical_volatility  # 2x historical volatility

    reward = 0  # Start with neutral reward

    # Assess recent return against threshold
    if recent_return < -threshold:
        reward -= 50  # Negative reward for high loss
    elif recent_return > threshold:
        reward += 50  # Positive reward for high gain

    # Add additional risk and trend assessments
    # Example: Consider RSI for overbought/oversold conditions
    rsi = enhanced_s[-2]  # Second to last feature is RSI
    if rsi > 70:
        reward -= 20  # Overbought condition
    elif rsi < 30:
        reward += 20  # Oversold condition

    return np.clip(reward, -100, 100)  # Ensure reward is in the range [-100, 100]