import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    features = []
    
    # Feature 1: 14-Day Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-14] if len(closing_prices) >= 14 else 0
    features.append(price_momentum)

    # Feature 2: 14-Day RSI
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return np.nan
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0).mean()
        loss = np.where(delta < 0, -delta, 0).mean()
        rs = gain / loss if loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)
    features.append(rsi if not np.isnan(rsi) else 50)  # Neutral when not enough data

    # Feature 3: MACD
    def compute_macd(prices):
        if len(prices) < 26:
            return np.nan, np.nan
        short_ema = np.mean(prices[-12:])  # Short window
        long_ema = np.mean(prices[-26:])   # Long window
        macd = short_ema - long_ema
        signal_line = np.mean(prices[-9:])  # Signal line
        return macd, signal_line

    macd, signal_line = compute_macd(closing_prices)
    features.append(macd if not np.isnan(macd) else 0)
    features.append(signal_line if not np.isnan(signal_line) else 0)

    # Feature 4: Average True Range (ATR)
    def compute_atr(prices, period=14):
        if len(prices) < period + 1:
            return np.nan
        high = np.max(prices[-period:])
        low = np.min(prices[-period:])
        return high - low

    atr = compute_atr(closing_prices)
    features.append(atr if not np.isnan(atr) else 0)

    # Feature 5: Z-score of Returns
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    z_score = (returns[-1] - np.mean(returns)) / np.std(returns) if len(returns) > 1 else 0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Calculate thresholds dynamically based on historical volatility
    historical_std = np.std(features) if len(features) > 0 else 1
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative reward for BUY-aligned features
        reward += 10 * features[2]  # Mild positive for selling if MACD is favorable
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += features[0] * 10  # Positive reward for momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += features[1] * 10  # Positive reward for MACD signal
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 15  # Buy signal
        elif features[1] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 15  # Sell signal
        else:
            reward -= 10  # Penalize breakout chasing
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds