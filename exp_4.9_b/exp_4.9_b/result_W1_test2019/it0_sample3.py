import numpy as np

def calculate_sma(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def calculate_ema(data, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0).mean()
    loss = -np.where(deltas < 0, deltas, 0).mean()
    rs = gain / loss if loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], 
                    np.maximum(np.abs(highs[1:] - closes[:-1]), 
                               np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:])

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # A. Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    price_vs_sma_5 = closing_prices[-1] / sma_5[-1] - 1 if len(sma_5) > 0 else 0
    price_vs_sma_20 = closing_prices[-1] / sma_20[-1] - 1 if len(sma_20) > 0 else 0
    
    trend_diff_5_10 = sma_5[-1] - sma_10[-1] if len(sma_5) > 0 and len(sma_10) > 0 else 0

    # B. Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # C. Volatility Indicators
    historical_vol_5 = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100
    historical_vol_20 = np.std(np.diff(closing_prices[-20:]) / closing_prices[-21:-1]) * 100 if len(closing_prices) > 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)
    
    # D. Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], 
                              np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))
    
    volume_price_correlation = np.corrcoef(volumes, closing_prices)[0, 1] if len(volumes) > 1 else 0
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 0

    # E. Market Regime Detection
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 > 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1] if len(closing_prices) > 1 else 0
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0

    # Compile the enhanced state
    enhanced_s = np.concatenate((
        s,
        np.array([
            sma_5[-1] if len(sma_5) > 0 else 0,
            sma_10[-1] if len(sma_10) > 0 else 0,
            sma_20[-1] if len(sma_20) > 0 else 0,
            ema_5[-1] if len(ema_5) > 0 else 0,
            ema_10[-1] if len(ema_10) > 0 else 0,
            price_vs_sma_5,
            price_vs_sma_20,
            trend_diff_5_10,
            rsi_5,
            rsi_10,
            rsi_14,
            historical_vol_5,
            historical_vol_20,
            atr,
            volume_price_correlation,
            volume_ratio,
            volatility_ratio,
            trend_strength,
            price_position,
            volume_ratio_regime,
            obv[-1] if len(obv) > 0 else 0
        ])
    ))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100  # Daily returns in percentage
    
    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol
    
    reward = 0
    
    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong BUY signal
            reward += 50
        elif recent_return < -threshold:  # Unfavorable condition
            reward -= 20
    
    elif position_flag == 1:  # Holding
        if recent_return < -threshold:  # Weakening trend, consider selling
            reward -= 30
        elif recent_return > 0:  # Positive return, encourage holding
            reward += 20
        else:  # Choppy market
            reward -= 10
    
    return np.clip(reward, -100, 100)