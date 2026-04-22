import numpy as np

def calculate_sma(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')

def calculate_ema(data, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean() if len(deltas) > 0 else 0
    loss = -np.where(deltas < 0, deltas, 0).mean() if len(deltas) > 0 else 0
    rs = gain / loss if loss != 0 else np.inf
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

def calculate_volatility(prices, window):
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * 100 if len(returns) >= window else np.nan  # Convert to percentage

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
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)

    # Volatility Indicators
    volatility_5 = calculate_volatility(closing_prices, 5)
    volatility_20 = calculate_volatility(closing_prices, 20)
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)

    # Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan

    # Market Regime Detection
    volatility_ratio = volatility_5 / volatility_20 if volatility_20 > 0 else np.nan
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] if len(closing_prices) > 1 else 0
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else np.nan
    
    # Create enhanced state with new features
    enhanced_s = np.concatenate((
        s,
        np.array([
            sma_5[-1] if len(sma_5) > 0 else 0,
            sma_10[-1] if len(sma_10) > 0 else 0,
            sma_20[-1] if len(sma_20) > 0 else 0,
            ema_5[-1] if len(ema_5) > 0 else 0,
            ema_10[-1] if len(ema_10) > 0 else 0,
            rsi_5,
            rsi_10,
            rsi_14,
            volatility_5,
            volatility_20,
            atr,
            obv[-1] if len(obv) > 0 else 0,
            volume_ratio,
            volatility_ratio,
            trend_strength,
            price_position,
            volume_ratio_regime
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
        if enhanced_s[120] > 1.05:  # Price above 5-day SMA
            reward += 50
        if enhanced_s[121] < 30:  # RSI < 30 (oversold)
            reward += 30
    else:  # Holding
        if enhanced_s[120] < 1.05:  # Price below 5-day SMA
            reward -= 50
        if enhanced_s[121] > 70:  # RSI > 70 (overbought)
            reward -= 30
        if recent_return < -threshold:  # Loss exceeding adaptive threshold
            reward -= 50
        
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]