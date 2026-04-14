import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    num_days = len(closing_prices)
    features = []

    # Feature 1: 14-Day Exponential Moving Average (EMA)
    def compute_ema(prices, span=14):
        if len(prices) < span:
            return np.nan
        weights = np.exp(np.linspace(-1., 0., span))
        weighted_prices = weights / weights.sum()
        ema = np.convolve(prices, weighted_prices, mode='valid')[-1]
        return ema
    
    ema = compute_ema(closing_prices)
    features.append(ema)

    # Feature 2: Average True Range (ATR) for volatility detection
    def compute_atr(prices, n=14):
        if len(prices) < n + 1:
            return np.nan
        high = prices[1::6][1:]  # High prices
        low = prices[2::6][1:]    # Low prices
        close = prices[0::6][:-1] # Closing prices
        tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
        return np.mean(tr[-n:])

    atr = compute_atr(s)
    features.append(atr)

    # Feature 3: Z-score of RSI to measure relative strength
    if num_days >= 15:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = np.abs(np.where(deltas < 0, deltas, 0)).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Z-score based on historical values
        historical_rsi = np.array([rsi])  # Replace with historical RSI values for more accurate Z-score
        rsi_mean = np.mean(historical_rsi)
        rsi_std = np.std(historical_rsi) if np.std(historical_rsi) != 0 else 1
        z_score_rsi = (rsi - rsi_mean) / rsi_std
    else:
        z_score_rsi = 0
    features.append(z_score_rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical standard deviation for dynamic thresholds
    historical_std = np.std(features) if np.std(features) != 0 else 1
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative for BUY signals
        reward += 10 * features[0]  # Mild positive for SELL based on EMA
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += 10 * features[0]  # Positive reward for EMA
        elif trend_direction < -0.3:  # Downtrend
            reward += 10 * features[2]  # Positive reward for Z-score RSI

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Assuming Z-score RSI < -1 indicates oversold
            reward += 15  # Buy signal
        elif features[2] > 1:  # Assuming Z-score RSI > 1 indicates overbought
            reward += 15  # Sell signal
        else:
            reward -= 10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds