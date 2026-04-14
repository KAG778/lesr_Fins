import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    num_days = len(closing_prices)
    
    features = []

    # Feature 1: 14-Day Momentum (Current Price - Price 14 days ago)
    momentum = closing_prices[-1] - closing_prices[-14] if num_days >= 14 else 0
    features.append(momentum)

    # Feature 2: 14-Day Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return np.nan
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = np.where(deltas < 0, -deltas, 0).mean()
        rs = gain / loss if loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)
    features.append(rsi if not np.isnan(rsi) else 50)  # Neutral RSI if insufficient data

    # Feature 3: Average True Range (ATR) for volatility detection
    def compute_atr(prices, n=14):
        if len(prices) < n + 1:
            return np.nan
        high = prices[1::6][1:]  # High prices
        low = prices[2::6][1:]   # Low prices
        close = prices[0::6][:-1]  # Previous closing prices
        tr = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
        return np.mean(tr[-n:])

    atr = compute_atr(s)
    features.append(atr if not np.isnan(atr) else 0)

    # Feature 4: Z-score of Returns
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

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(features) if len(features) > 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10 * features[2]  # Mild positive for selling if ATR is favorable
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