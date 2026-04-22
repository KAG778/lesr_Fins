import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(prices.shape)
    ema[window-1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-window:]) if len(gains) >= window else 0
    avg_loss = np.mean(losses[-window:]) if len(losses) >= window else 0
    
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                   np.maximum(np.abs(highs[1:] - closes[:-1]), 
                              np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def calculate_volatility(prices):
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * 100  # Convert to percentage

def calculate_obv(prices, volumes):
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif prices[i] < prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    return obv

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]

    # Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else 0
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else 0
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else 0
    ema_5 = calculate_ema(closing_prices, 5)[-1] if len(closing_prices) >= 5 else 0
    ema_10 = calculate_ema(closing_prices, 10)[-1] if len(closing_prices) >= 10 else 0

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5) if len(closing_prices) >= 5 else np.nan
    rsi_10 = calculate_rsi(closing_prices, 10) if len(closing_prices) >= 10 else np.nan
    rsi_14 = calculate_rsi(closing_prices, 14) if len(closing_prices) >= 14 else np.nan
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if len(closing_prices) > 1 else 0

    # Volatility Indicators
    historical_volatility = calculate_volatility(closing_prices)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0

    # Market Regime Detection
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else 0
    volatility_ratio = historical_volatility / (np.std(np.diff(closing_prices[-20:])) * 100) if len(closing_prices) > 20 else 0
    trend_strength = np.corrcoef(np.arange(len(closing_prices)), closing_prices)[0, 1] if len(closing_prices) > 1 else 0

    # Creating enhanced state with new features
    enhanced_s = np.concatenate((
        s,
        np.array([
            sma_5, sma_10, sma_20,
            ema_5, ema_10,
            rsi_5, rsi_10, rsi_14,
            historical_volatility, atr,
            obv, volume_ratio,
            price_position, volatility_ratio,
            trend_strength,
            momentum,
            closing_prices[-1] / sma_5 if sma_5 > 0 else 0,  # Price/SMA
            closing_prices[-1] / sma_10 if sma_10 > 0 else 0,  # Price/SMA
            closing_prices[-1] / sma_20 if sma_20 > 0 else 0,  # Price/SMA
            (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100,  # Daily return
            np.mean(closing_prices[-5:]),  # 5-day average price
            np.mean(closing_prices[-10:]),  # 10-day average price
            np.mean(closing_prices[-20:]),  # 20-day average price
            np.std(closing_prices[-5:]),  # 5-day volatility
            np.std(closing_prices[-10:]),  # 10-day volatility
            np.std(closing_prices[-20:]),  # 20-day volatility
            np.mean(volumes[-5:]),  # 5-day average volume
            np.mean(volumes[-10:]),  # 10-day average volume
            np.mean(volumes[-20:]),  # 20-day average volume
            np.std(volumes[-5:]),  # 5-day volume volatility
            np.std(volumes[-10:]),  # 10-day volume volatility
            np.std(volumes[-20:]),  # 20-day volume volatility
        ])
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Daily return
    historical_volatility = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) > 1 else 1  # Historical volatility
    threshold = 2 * historical_volatility  # Use volatility-adaptive threshold

    reward = 0
    
    if position == 0:  # Not holding
        if enhanced_s[120] < 30:  # RSI < 30 (oversold)
            reward += 50
        if enhanced_s[121] > 0.1:  # Trend strength condition for buy signal
            reward += 10
    else:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50
        elif recent_return > 0:  # Positive return
            reward += 30
        if enhanced_s[120] > 70:  # RSI > 70 (overbought)
            reward -= 10
        if enhanced_s[121] < 0.1:  # Weak trend signal
            reward -= 10
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]