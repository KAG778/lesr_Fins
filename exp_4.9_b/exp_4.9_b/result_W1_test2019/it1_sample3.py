import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window) / window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(prices.shape)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def compute_obv(prices, volumes):
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif prices[i] < prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    return obv[-1]

def compute_trend_strength(prices):
    from sklearn.linear_model import LinearRegression
    x = np.arange(len(prices)).reshape(-1, 1)
    y = prices.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.score(x, y)

def revise_state(s):
    closing_prices = s[0:20]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Trend indicators
    sma_5 = calculate_sma(closing_prices, 5)[-1] if len(closing_prices) >= 5 else np.nan
    sma_10 = calculate_sma(closing_prices, 10)[-1] if len(closing_prices) >= 10 else np.nan
    sma_20 = calculate_sma(closing_prices, 20)[-1] if len(closing_prices) >= 20 else np.nan
    ema_5 = calculate_ema(closing_prices, 5)[-1]
    ema_10 = calculate_ema(closing_prices, 10)[-1]
    
    price_vs_sma_5 = closing_prices[-1] - sma_5
    price_vs_sma_10 = closing_prices[-1] - sma_10
    price_vs_sma_20 = closing_prices[-1] - sma_20
    
    # Momentum indicators
    rsi_14 = calculate_rsi(closing_prices, 14)
    macd = ema_5 - ema_10  # Simplified MACD
    
    # Volatility indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:]) / closing_prices[-6:-1]) * 100 if len(closing_prices) >= 6 else np.nan
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100 if len(closing_prices) >= 21 else np.nan
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # Volume indicators
    obv = compute_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan
    
    # Market regime detection
    trend_strength = compute_trend_strength(closing_prices)
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 > 0 else np.nan
    
    # Creating the enhanced state with new features
    enhanced_s = np.concatenate([
        s,
        np.array([sma_5, sma_10, sma_20, ema_5, ema_10,
                  price_vs_sma_5, price_vs_sma_10, price_vs_sma_20,
                  rsi_14, macd,
                  historical_volatility_5, historical_volatility_20, atr,
                  obv, volume_ratio,
                  trend_strength, price_position, volatility_ratio])
    ])
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    
    # Calculate returns and historical volatility
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100 if len(closing_prices) >= 2 else 1  # Avoid division by zero
    threshold = 2 * historical_vol  # Volatility-adaptive threshold
    
    reward = 0
    if position == 0:  # Not holding
        if enhanced_s[120] > 0 and enhanced_s[121] < 30:  # RSI condition for buy signal
            reward += 50  # Strong buy signal
        elif enhanced_s[120] < 0:  # Trend weakens
            reward -= 20  # Unfavorable condition
    else:  # Holding
        if recent_return < -threshold:  # Significant loss
            reward -= 50  # Exit signal
        elif recent_return > 0:  # Positive return
            reward += 20  # Encourage holding
        else:  # Choppy market
            reward -= 10  # Penalize
    
    return np.clip(reward, -100, 100)  # Limit reward to range [-100, 100]