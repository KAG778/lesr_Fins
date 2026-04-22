import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast_window=12, slow_window=26, signal_window=9):
    ema_fast = calculate_ema(prices, fast_window)
    ema_slow = calculate_ema(prices, slow_window)
    macd = ema_fast[-1] - ema_slow[-1]
    signal = calculate_ema(prices[-signal_window:], signal_window)
    return macd, signal

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-window:])
    return atr

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[20:40]
    low_prices = s[40:60]
    volumes = s[60:80]

    enhanced_s = np.copy(s)

    # Calculate indicators
    # 5-day SMA
    sma_5 = calculate_sma(closing_prices, 5)
    enhanced_s = np.concatenate((enhanced_s, sma_5[-1:] if len(sma_5) > 0 else np.array([np.nan])))

    # 10-day EMA
    ema_10 = calculate_ema(closing_prices, 10)
    enhanced_s = np.concatenate((enhanced_s, ema_10[-1:] if len(ema_10) > 0 else np.array([np.nan])))

    # 14-day RSI
    rsi = calculate_rsi(closing_prices, 14)
    enhanced_s = np.concatenate((enhanced_s, np.array([rsi])))

    # MACD
    macd, signal = calculate_macd(closing_prices)
    enhanced_s = np.concatenate((enhanced_s, np.array([macd, signal])))

    # Average True Range (ATR)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    enhanced_s = np.concatenate((enhanced_s, np.array([atr])))

    # 20-day Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    volatility = np.std(returns) if len(returns) > 0 else 0
    enhanced_s = np.concatenate((enhanced_s, np.array([volatility])))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(historical_returns) if len(historical_returns) > 0 else 0

    threshold = 2 * historical_vol  # Volatility-adaptive threshold
    reward = 0

    # Reward conditions
    if recent_return > threshold:
        reward += 50  # Strong upward movement
    elif recent_return < -threshold:
        reward -= 50  # Strong downward movement
    else:
        reward += 10  # Mild return

    # Trend checking using moving averages
    if enhanced_s[-6] > enhanced_s[-7]:  # Last two features: MACD and Signal
        reward += 20  # Positive momentum
    elif enhanced_s[-6] < enhanced_s[-7]:
        reward -= 20  # Negative momentum

    # Risk assessment based on volatility
    if historical_vol > 3:  # High volatility threshold
        reward -= 20  # Penalize riskier conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range