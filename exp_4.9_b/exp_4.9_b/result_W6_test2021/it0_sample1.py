import numpy as np

def revise_state(s):
    # Extracting the last 20 days of closing prices for calculations
    closing_prices = s[0:20]
    opening_prices = s[20:39]
    high_prices = s[40:59]
    low_prices = s[60:79]
    volumes = s[80:99]
    
    # Calculating daily returns
    returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Feature Calculation
    features = []
    
    # A. Multi-timeframe Trend Indicators
    # Moving Averages
    sma_5 = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) == 5 else np.nan
    sma_10 = np.mean(closing_prices[-10:]) if len(closing_prices[-10:]) == 10 else np.nan
    sma_20 = np.mean(closing_prices[-20:]) if len(closing_prices[-20:]) == 20 else np.nan
    features.extend([sma_5, sma_10, sma_20])
    
    # Price relative to moving averages
    price_vs_sma_5 = closing_prices[-1] - sma_5 if not np.isnan(sma_5) else np.nan
    price_vs_sma_10 = closing_prices[-1] - sma_10 if not np.isnan(sma_10) else np.nan
    price_vs_sma_20 = closing_prices[-1] - sma_20 if not np.isnan(sma_20) else np.nan
    features.extend([price_vs_sma_5, price_vs_sma_10, price_vs_sma_20])
    
    # B. Momentum Indicators
    rsi_14 = calculate_rsi(returns, 14)
    features.append(rsi_14)
    
    # C. Volatility Indicators
    hist_vol_5 = np.std(returns[-5:]) * 100 if len(returns[-5:]) == 5 else np.nan
    hist_vol_20 = np.std(returns[-20:]) * 100 if len(returns[-20:]) == 20 else np.nan
    features.extend([hist_vol_5, hist_vol_20])
    
    # D. Volume-Price Relationship
    obv = calculate_obv(volumes, closing_prices)
    features.append(obv)
    
    # E. Market Regime Detection
    volatility_ratio = hist_vol_5 / hist_vol_20 if hist_vol_20 != 0 else np.nan
    trend_strength = calculate_trend_strength(closing_prices)
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if np.max(closing_prices[-20:]) != np.min(closing_prices[-20:]) else np.nan
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan
    
    features.extend([volatility_ratio, trend_strength, price_position, volume_ratio_regime])
    
    # Combine the original state with the new features
    enhanced_s = np.concatenate((s, np.array(features)))
    
    return enhanced_s

def calculate_rsi(returns, period):
    if len(returns) < period:
        return np.nan
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(volumes, closing_prices):
    obv = np.zeros_like(closing_prices)
    for i in range(1, len(closing_prices)):
        if closing_prices[i] > closing_prices[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closing_prices[i] < closing_prices[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv[-1]

def calculate_trend_strength(prices):
    from sklearn.linear_model import LinearRegression
    x = np.arange(len(prices)).reshape(-1, 1)  # Days as a feature
    y = prices.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    r_squared = model.score(x, y)
    return r_squared

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    returns = np.diff(closing_prices) / closing_prices[:-1]
    
    # Calculate historical volatility
    historical_vol = np.std(returns) * 100  # Convert to percentage
    threshold = 2 * historical_vol  # Adapting the threshold to volatility
    
    recent_return = returns[-1] * 100 if len(returns) > 0 else 0  # Latest return in percentage
    reward = 0
    
    if position_flag == 0.0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    elif position_flag == 1.0:  # Holding
        if recent_return < -threshold:  # Weakening trend
            reward -= 50
        elif recent_return > 0:  # Continue holding
            reward += 20
    
    # Penalize uncertain market conditions
    if np.isnan(recent_return) or np.std(returns) < (0.5 * historical_vol):
        reward -= 20  # Choppy market

    return np.clip(reward, -100, 100)