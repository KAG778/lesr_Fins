import numpy as np

def revise_state(s):
    # Extract the closing prices and volumes
    closing_prices = s[0:20]
    opening_prices = s[20:40]
    high_prices = s[40:60]
    low_prices = s[60:80]
    volumes = s[80:100]
    
    # Feature arrays
    features = []

    # A. Multi-timeframe Trend Indicators
    # 5-day SMA
    sma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    features.append(closing_prices[-1] - sma_5)  # Price relative to 5-day SMA

    # 10-day SMA
    sma_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan
    features.append(closing_prices[-1] - sma_10)  # Price relative to 10-day SMA
    
    # 20-day SMA
    sma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else np.nan
    features.append(closing_prices[-1] - sma_20)  # Price relative to 20-day SMA

    # Short vs Long MA difference (5-day vs 20-day)
    features.append(sma_5 - sma_20 if not np.isnan(sma_5) and not np.isnan(sma_20) else 0)

    # B. Momentum Indicators
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = -np.where(deltas < 0, deltas, 0).mean()
        rs = gain / loss if loss != 0 else np.nan
        return 100 - (100 / (1 + rs))

    # 5-day RSI
    rsi_5 = calculate_rsi(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    features.append(rsi_5)

    # 10-day RSI
    rsi_10 = calculate_rsi(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan
    features.append(rsi_10)

    # 14-day RSI
    rsi_14 = calculate_rsi(closing_prices[-14:]) if len(closing_prices) >= 14 else np.nan
    features.append(rsi_14)

    # MACD: Difference between short and long EMA
    def calculate_macd(prices):
        ema_12 = np.mean(prices[-12:]) if len(prices) >= 12 else np.nan
        ema_26 = np.mean(prices[-26:]) if len(prices) >= 26 else np.nan
        return ema_12 - ema_26 if not np.isnan(ema_12) and not np.isnan(ema_26) else np.nan

    macd = calculate_macd(closing_prices)
    features.append(macd)

    # C. Volatility Indicators
    def calculate_historical_volatility(prices, window=20):
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) if len(returns) >= 2 else np.nan

    # 5-day Historical Volatility
    hist_vol_5 = calculate_historical_volatility(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    features.append(hist_vol_5)

    # 20-day Historical Volatility
    hist_vol_20 = calculate_historical_volatility(closing_prices[-20:]) if len(closing_prices) >= 20 else np.nan
    features.append(hist_vol_20)

    # D. Volume-Price Relationship
    def calculate_obv(prices, volumes):
        obv = np.zeros_like(prices)
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif prices[i] < prices[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]
        return obv[-1] if len(obv) > 0 else np.nan

    obv = calculate_obv(closing_prices, volumes)
    features.append(obv)

    # Volume-Price correlation
    volume_price_corr = np.corrcoef(closing_prices, volumes)[0, 1] if len(closing_prices) > 1 else np.nan
    features.append(volume_price_corr)

    # E. Market Regime Detection
    # Volatility Ratio
    volatility_ratio = hist_vol_5 / hist_vol_20 if hist_vol_20 != 0 else np.nan
    features.append(volatility_ratio)

    # Trend Strength (R² of linear regression)
    if len(closing_prices) >= 20:
        from sklearn.linear_model import LinearRegression
        x = np.arange(20).reshape(-1, 1)
        y = closing_prices[-20:].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        trend_strength = model.score(x, y)
    else:
        trend_strength = np.nan
    features.append(trend_strength)

    # Price Position in 20-day range
    price_position = (closing_prices[-1] - np.min(closing_prices[-20:])) / (np.max(closing_prices[-20:]) - np.min(closing_prices[-20:])) if len(closing_prices) >= 20 else np.nan
    features.append(price_position)

    # Volume Ratio Regime
    volume_ratio_regime = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) != 0 else np.nan
    features.append(volume_ratio_regime)

    # Combine original state with new features
    enhanced_s = np.concatenate((s, np.array(features)))

    return enhanced_s

def intrinsic_reward(enhanced_s):
    position_flag = enhanced_s[-1]
    closing_prices = enhanced_s[0:20]
    recent_return = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100  # Calculate daily return

    # Calculate historical volatility from closing prices
    historical_vol = np.std(np.diff(closing_prices) / closing_prices[:-1] * 100)  # e.g., TSLA~4%
    
    # Use 2x historical volatility as threshold
    threshold = 2 * historical_vol  # TSLA: ~8%, MSFT: ~4%
    
    reward = 0

    if position_flag == 0:  # Not holding
        if recent_return > threshold:  # Strong uptrend
            reward += 50
        elif recent_return < -threshold:  # Strong downtrend
            reward -= 50
    elif position_flag == 1:  # Holding
        if recent_return > 0:  # Positive return
            reward += 10
        if recent_return < -threshold:  # Negative return
            reward -= 50

    # Penalize uncertain/choppy market conditions
    volatility_ratio = enhanced_s[100]  # Assuming it's the 100th feature
    if volatility_ratio <= 2.0:  # Not extreme volatility
        reward -= 10

    return np.clip(reward, -100, 100)