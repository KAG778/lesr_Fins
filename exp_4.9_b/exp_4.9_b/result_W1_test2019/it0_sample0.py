import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window), 'valid') / window

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(prices.shape)
    ema[window-1] = np.mean(prices[:window])
    for i in range(window, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i-1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    
    avg_gain = np.mean(gain[-window:]) if len(gain) >= window else 0
    avg_loss = np.mean(loss[-window:]) if len(loss) >= window else 0
    
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_volatility(prices, window):
    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * 100  # Convert to percentage

def calculate_obv(close_prices, volume):
    obv = np.zeros(close_prices.shape)
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close_prices[i] < close_prices[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv

def revise_state(raw_state):
    closing_prices = raw_state[0:20]
    opening_prices = raw_state[20:40]
    high_prices = raw_state[40:60]
    low_prices = raw_state[60:80]
    volumes = raw_state[80:100]
    
    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    
    # Calculate current price relative to SMAs
    price_to_sma_5 = closing_prices[-1] / sma_5[-1] if len(sma_5) > 0 else 0
    price_to_sma_10 = closing_prices[-1] / sma_10[-1] if len(sma_10) > 0 else 0
    price_to_sma_20 = closing_prices[-1] / sma_20[-1] if len(sma_20) > 0 else 0
    
    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Volatility Indicators
    volatility_5 = calculate_volatility(closing_prices, 5)
    volatility_20 = calculate_volatility(closing_prices, 20)
    
    # Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    
    # Market Regime Detection
    volatility_ratio = volatility_5 / volatility_20 if volatility_20 > 0 else 0
    trend_strength = np.polyfit(range(len(closing_prices)), closing_prices, 1)[0]  # Simple slope as trend strength
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0
    
    # Compile enhanced state
    enhanced_s = np.concatenate((
        raw_state,
        np.array([
            sma_5[-1] if len(sma_5) > 0 else 0,
            sma_10[-1] if len(sma_10) > 0 else 0,
            sma_20[-1] if len(sma_20) > 0 else 0,
            price_to_sma_5,
            price_to_sma_10,
            price_to_sma_20,
            rsi_5,
            rsi_10,
            rsi_14,
            volatility_5,
            volatility_20,
            obv[-1] if len(obv) > 0 else 0,
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
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1]) * 100  # Historical volatility
    
    reward = 0
    
    if position == 0:  # Not holding
        if enhanced_s[120] > 1.05:  # Price above 5-day SMA
            reward += 50
        if enhanced_s[122] < 30:  # RSI < 30 (oversold)
            reward += 30
    else:  # Holding
        if enhanced_s[120] < 1.05:  # Price below 5-day SMA
            reward -= 50
        if enhanced_s[121] > 70:  # RSI > 70 (overbought)
            reward -= 30
        if recent_return < -2 * historical_vol:  # Loss exceeding 2x historical volatility
            reward -= 50
        
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]