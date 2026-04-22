import numpy as np

def calculate_sma(prices, window):
    return np.mean(prices[-window:]) if len(prices) >= window else np.nan

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

def calculate_atr(highs, lows, closes, window):
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-window:]) if len(tr) >= window else np.nan

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

    enhanced_features = []

    # A. Multi-timeframe Trend Indicators
    enhanced_features.append(closing_prices[-1] - calculate_sma(closing_prices, 5))  # Price vs 5-day SMA
    enhanced_features.append(closing_prices[-1] - calculate_sma(closing_prices, 10))  # Price vs 10-day SMA
    enhanced_features.append(closing_prices[-1] - calculate_sma(closing_prices, 20))  # Price vs 20-day SMA
    enhanced_features.append(calculate_ema(closing_prices, 5))  # 5-day EMA
    enhanced_features.append(calculate_ema(closing_prices, 10))  # 10-day EMA
    enhanced_features.append(calculate_ema(closing_prices, 20))  # 20-day EMA
    
    # B. Momentum Indicators
    enhanced_features.append(calculate_rsi(closing_prices, 5))  # 5-day RSI
    enhanced_features.append(calculate_rsi(closing_prices, 10))  # 10-day RSI
    enhanced_features.append(calculate_rsi(closing_prices, 14))  # 14-day RSI

    # C. Volatility Indicators
    historical_volatility_5 = np.std(np.diff(closing_prices[-5:])) if len(closing_prices) >= 5 else 0
    historical_volatility_20 = np.std(np.diff(closing_prices[-20:])) if len(closing_prices) >= 20 else 0
    atr = calculate_atr(high_prices, lows, closing_prices, 14)
    enhanced_features.append(historical_volatility_5)
    enhanced_features.append(historical_volatility_20)
    enhanced_features.append(atr)

    # D. Volume-Price Relationship
    obv = calculate_obv(closing_prices, volumes)
    enhanced_features.append(obv)
    enhanced_features.append(np.mean(volumes[-5:]) / np.mean(volumes[-20:]))  # Volume ratio

    # E. Market Regime Detection
    volatility_ratio = historical_volatility_5 / historical_volatility_20 if historical_volatility_20 > 0 else 0
    trend_strength = np.corrcoef(range(len(closing_prices)), closing_prices)[0, 1]  # R² of linear regression
    price_position = (closing_prices[-1] - np.min(closing_prices)) / (np.max(closing_prices) - np.min(closing_prices)) if np.max(closing_prices) != np.min(closing_prices) else 0
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0

    enhanced_features.append(volatility_ratio)
    enhanced_features.append(trend_strength)
    enhanced_features.append(price_position)
    enhanced_features.append(volume_ratio_regime)

    # Combine original state and new features
    enhanced_s = np.concatenate((s, np.array(enhanced_features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]  # Position flag
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Calculate daily return

    # Calculate historical volatility from closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1] * 100
    historical_vol = np.std(daily_returns) if len(daily_returns) > 0 else 0
    threshold = 2 * historical_vol  # Adaptive threshold

    reward = 0

    if position_flag == 0:  # Not holding
        if enhanced_s[121] > 0:  # Assuming trend_strength is at index 121
            reward += 50  # Positive reward for strong uptrend
        if recent_return > threshold:
            reward += 30  # Reward for high returns
        else:
            reward -= 20  # Penalize for not taking action in favorable conditions

    elif position_flag == 1:  # Holding
        if enhanced_s[121] < 0:  # Trend weakening
            reward -= 50  # Negative reward for selling signal
        else:
            reward += 20  # Reward for holding during uptrend

    # Penalize uncertain market conditions
    if enhanced_s[122] < 0.5:  # Assuming volatility_ratio at index 122
        reward -= 20  # Penalize for holding in choppy conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]