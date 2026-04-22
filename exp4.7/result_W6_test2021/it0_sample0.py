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
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else np.mean(gain)
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else np.mean(loss)
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculate SMA, EMA, and RSI for different windows
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)

    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    rsi_14 = calculate_rsi(closing_prices, 14)

    # Create enhanced state
    enhanced_s = np.concatenate([
        s,
        np.pad(sma_5, (4, 0), 'constant', constant_values=np.nan),
        np.pad(sma_10, (9, 0), 'constant', constant_values=np.nan),
        np.pad(sma_20, (19, 0), 'constant', constant_values=np.nan),
        np.pad(ema_5, (4, 0), 'constant', constant_values=np.nan),
        np.pad(ema_10, (9, 0), 'constant', constant_values=np.nan),
        np.pad(np.array([rsi_14]), (13, 0), 'constant', constant_values=np.nan)
    ])

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Percentage return

    # Calculate historical volatility
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns)

    # Define reward
    reward = 0
    
    # Volatility-adaptive thresholds
    threshold = 2 * historical_vol

    # Conditions for reward adjustment
    if recent_return > threshold:
        reward += 50  # Strong positive signal
    elif recent_return < -threshold:
        reward -= 50  # Strong negative signal
    
    # Add additional conditions based on RSI for trading viability
    rsi = enhanced_s[119]  # Assuming the last feature is RSI
    if rsi < 30:
        reward += 30  # Oversold condition
    elif rsi > 70:
        reward -= 30  # Overbought condition

    return np.clip(reward, -100, 100)  # Ensure reward is within range