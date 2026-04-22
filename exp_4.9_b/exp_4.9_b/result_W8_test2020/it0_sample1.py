import numpy as np

def calculate_sma(prices, window):
    return np.convolve(prices, np.ones(window)/window, mode='valid')

def calculate_ema(prices, window):
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * alpha) + (ema[i - 1] * (1 - alpha))
    return ema

def calculate_rsi(prices, window):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-window:]) if len(gain[-window:]) > 0 else 0
    avg_loss = np.mean(loss[-window:]) if len(loss[-window:]) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    return rsi

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Multi-timeframe Trend Indicators
    sma_5 = calculate_sma(closing_prices, 5)
    sma_10 = calculate_sma(closing_prices, 10)
    sma_20 = calculate_sma(closing_prices, 20)
    ema_5 = calculate_ema(closing_prices, 5)[-20:]  # Use only the latest 20 days
    ema_10 = calculate_ema(closing_prices, 10)[-20:]
    
    trend_strength = np.corrcoef(closing_prices[-20:], range(len(closing_prices[-20:])))[:-1, -1][0]
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) > np.min(closing_prices[-20:]) else 0
    
    # Momentum Indicators
    rsi_5 = calculate_rsi(closing_prices, 5)
    rsi_10 = calculate_rsi(closing_prices, 10)
    rsi_14 = calculate_rsi(closing_prices, 14)
    
    # Volatility Indicators
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_vol_5 = np.std(daily_returns[-5:]) * 100
    historical_vol_20 = np.std(daily_returns[-20:]) * 100
    volatility_ratio = historical_vol_5 / historical_vol_20 if historical_vol_20 > 0 else 0
    
    # Volume-Price Relationship
    obv = np.cumsum(np.where(np.diff(closing_prices) > 0, volumes[1:], -volumes[1:]))
    volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0
    
    # Market Regime Detection
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0

    # Collecting features
    enhanced_s = np.concatenate((
        s,  # Original state
        [sma_5[-1], sma_10[-1], sma_20[-1], ema_5[-1], ema_10[-1], trend_strength, price_position,
         rsi_5, rsi_10, rsi_14, historical_vol_5, historical_vol_20, volatility_ratio,
         obv[-1], volume_ratio, volume_ratio_regime]
    ))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    
    # Calculate recent return
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_vol = np.std(daily_returns) * 100

    # Use relative thresholds for rewards
    threshold = 2 * historical_vol  # Adaptive threshold based on stock volatility
    
    reward = 0
    
    if position == 0:  # Not holding
        if recent_return > threshold:  # Strong buy signal
            reward += 50
        elif recent_return < -threshold:  # Strong sell signal
            reward -= 50
    else:  # Holding
        if recent_return < -threshold:  # Consider selling
            reward -= 50
        elif recent_return > threshold:  # Consider holding during uptrend
            reward += 50
    
    return reward