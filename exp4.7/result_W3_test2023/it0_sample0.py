import numpy as np

def moving_average(prices, window):
    """Calculate moving average."""
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = moving_average(gain, window)
    avg_loss = moving_average(loss, window)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate((np.full(window-1, np.nan), rsi))  # Fill with NaN for the initial values

def calculate_atr(highs, lows, closes, window=14):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    atr = moving_average(tr, window)
    return np.concatenate((np.full(window-1, np.nan), atr))

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Calculate additional technical indicators
    sma_5 = moving_average(closing_prices, 5)
    sma_10 = moving_average(closing_prices, 10)
    sma_20 = moving_average(closing_prices, 20)
    rsi = calculate_rsi(closing_prices, window=14)
    atr = calculate_atr(high_prices, low_prices, closing_prices, window=14)

    # Handle edge cases: pad the beginning with NaN where necessary
    enhanced_s = np.concatenate([
        closing_prices,
        opening_prices,
        high_prices,
        low_prices,
        volumes,
        adjusted_closing_prices,
        sma_5,
        sma_10,
        sma_20,
        rsi,
        atr
    ])

    # Fill NaN values with zeros or some other strategy if necessary
    enhanced_s = np.nan_to_num(enhanced_s)

    return enhanced_s

def intrinsic_reward(enhanced_s):
    # Assume the last closing price is the most recent
    recent_close = enhanced_s[0]
    previous_close = enhanced_s[1]  # Immediate previous closing price
    recent_return = (recent_close - previous_close) / previous_close * 100  # in percentage

    # Historical volatility (calculated from the last 20 closing prices)
    historical_returns = np.diff(enhanced_s[0:20]) / enhanced_s[0:19] * 100
    historical_vol = np.std(historical_returns)

    # Use a relative threshold
    threshold = 2 * historical_vol  # Adaptive threshold based on historical volatility

    reward = 0

    # Reward structure
    if recent_return > threshold:
        reward += 50  # Positive momentum
    elif recent_return < -threshold:
        reward -= 50  # Negative momentum
        
    # Risk control based on ATR (if included in the enhanced state)
    atr = enhanced_s[100]  # Assuming ATR is at position 100 in enhanced state
    if atr > 0 and (recent_return / atr) < -1:  # Risky situation
        reward -= 50

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range