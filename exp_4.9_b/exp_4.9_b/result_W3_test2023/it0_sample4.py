import numpy as np

def calculate_sma(prices, window):
    if len(prices) < window:
        return np.nan
    return np.mean(prices[-window:])

def calculate_ema(prices, window):
    if len(prices) < window:
        return np.nan
    alpha = 2 / (window + 1)
    ema = prices[-window]
    for price in prices[-window:]:
        ema = (price - ema) * alpha + ema
    return ema

def calculate_rsi(prices, window):
    if len(prices) < window:
        return np.nan
    deltas = np.diff(prices)
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else np.nan
    return 100 - (100 / (1 + rs))

def calculate_volatility(prices, window):
    if len(prices) < window:
        return np.nan
    return np.std(np.diff(prices) / prices[:-1] * 100)

def calculate_obv(prices, volumes):
    if len(prices) < 2:
        return 0
    obv = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            obv += volumes[i]
        elif prices[i] < prices[i - 1]:
            obv -= volumes[i]
    return obv

def revise_state(s):
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    enhanced_s = list(s)  # Start with the original state
    
    # Trend Indicators
    enhanced_s.append(calculate_sma(closing_prices, 5))
    enhanced_s.append(calculate_sma(closing_prices, 10))
    enhanced_s.append(calculate_sma(closing_prices, 20))
    enhanced_s.append(calculate_ema(closing_prices, 5))
    enhanced_s.append(calculate_ema(closing_prices, 10))
    enhanced_s.append(calculate_ema(closing_prices, 20))
    enhanced_s.append(closing_prices[-1] - calculate_sma(closing_prices, 10))  # Price above 10-day SMA
    enhanced_s.append(closing_prices[-1] - calculate_sma(closing_prices, 20))  # Price above 20-day SMA

    # Momentum Indicators
    enhanced_s.append(calculate_rsi(closing_prices, 5))
    enhanced_s.append(calculate_rsi(closing_prices, 10))
    enhanced_s.append(calculate_rsi(closing_prices, 14))
    enhanced_s.append(calculate_volatility(closing_prices, 5))  # 5-day volatility
    enhanced_s.append(calculate_volatility(closing_prices, 20)) # 20-day volatility
    
    # Volatility Indicators
    enhanced_s.append(calculate_volatility(closing_prices, 5))
    enhanced_s.append(calculate_volatility(closing_prices, 20))
    
    # Volume Indicators
    obv = calculate_obv(closing_prices, volumes)
    enhanced_s.append(obv)
    enhanced_s.append(np.mean(volumes[-5:]) / np.mean(volumes[-20:]))  # Volume ratio
    
    # Market Regime Detection
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(daily_returns)
    enhanced_s.append(calculate_volatility(closing_prices, 5) / calculate_volatility(closing_prices, 20))  # Volatility ratio
    trend_strength = np.polyfit(range(len(closing_prices)), closing_prices, 1)[0]  # Linear regression slope
    enhanced_s.append(trend_strength)
    enhanced_s.append((closing_prices[-1] - min(closing_prices)) / (max(closing_prices) - min(closing_prices)))  # Price position in range
    enhanced_s.append(np.mean(volumes[-5:]) / np.mean(volumes[-20:]))  # Volume ratio regime
    
    return np.array(enhanced_s)

def intrinsic_reward(enhanced_s):
    position = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100
    
    # Calculate historical volatility
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(daily_returns)
    threshold = 2 * historical_vol  # Adaptive threshold for high volatility
    
    reward = 0
    
    if position == 0:  # Not holding
        if enhanced_s[9] < 30:  # RSI 5-day
            reward += 50  # Oversold condition
        if enhanced_s[12] > 0:  # Trend strength positive
            reward += 50  # Clear buy signal
    else:  # Holding
        if enhanced_s[9] > 70:  # RSI 5-day
            reward -= 50  # Overbought condition, consider selling
        if recent_return < -threshold:  # Check for significant drop
            reward -= 50
    
    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]