import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros(len(prices))
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
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
    return np.convolve(tr, np.ones(window)/window, mode='valid')

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    adjusted_closing_prices = s[100:120]

    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)
    ema_10 = calculate_ema(closing_prices, 10)
    
    trend_short_long_diff = sma_5[-1] - sma_20[-1] if len(sma_20) > 0 else 0
    price_vs_sma_20 = closing_prices[-1] - sma_20[-1] if len(sma_20) > 0 else 0

    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    
    # Volatility Indicators
    volatility_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) > 5 else 0
    volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) > 20 else 0
    atr = calculate_atr(high_prices, low_prices, closing_prices, 14)[-1] if len(high_prices) > 14 else 0

    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], 
                             np.where(np.diff(closing_prices) < 0, -volumes[1:], 0)))
    volume_ratio = volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 0

    # Market Regime Detection
    volatility_ratio = volatility_5 / volatility_20 if volatility_20 > 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # Simple linear regression R²
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = (np.mean(volumes[-5:]) / np.mean(volumes[-20:])) if np.mean(volumes[-20:]) > 0 else 0
    
    # Create enhanced state with new features
    enhanced_s = np.concatenate((s, 
                                  [sma_5[-1], sma_10[-1], sma_20[-1], 
                                   ema_5[-1], ema_10[-1],
                                   trend_short_long_diff, price_vs_sma_20,
                                   rsi_5, rsi_10, rsi_14, momentum,
                                   volatility_5, volatility_20, atr,
                                   obv[-1], volume_ratio,
                                   volatility_ratio, trend_strength, 
                                   price_position, volume_ratio_regime]))
    
    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Historical volatility for threshold calculation
    returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(returns) if len(returns) > 0 else 0
    threshold = 2 * historical_vol  # Adaptive threshold
    
    reward = 0
    
    if position == 0:  # Not holding
        if enhanced_s[120] > 0:  # Assuming trend_strength is at index 120
            reward += 50  # Positive reward for strong uptrend
        if recent_return > threshold:
            reward += 30  # Reward for high returns
        else:
            reward -= 20  # Penalize for not taking action in favorable conditions

    elif position == 1:  # Holding
        if enhanced_s[120] < 0:  # Trend weakening
            reward -= 50  # Negative reward for selling signal
        else:
            reward += 20  # Reward for holding during uptrend
    
    # Penalize uncertain market conditions
    if enhanced_s[121] < 0.5:  # Assuming trend_strength at index 121
        reward -= 20  # Penalize for holding in choppy conditions
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]