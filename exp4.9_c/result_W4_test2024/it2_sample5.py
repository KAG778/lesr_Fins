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
    features.append(ema if not np.isnan(ema) else 0)

    # Feature 2: Average True Range (ATR)
    def compute_atr(prices, n=14):
        if len(prices) < n + 1:
            return np.nan
        high = s[1::6][1:]  # High prices
        low = s[2::6][1:]   # Low prices
        close = closing_prices[:-1]  # Closing prices
        tr = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
        return np.mean(tr[-n:])

    atr = compute_atr(s)
    features.append(atr if not np.isnan(atr) else 0)

    # Feature 3: Z-score of Daily Returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    z_score_returns = (daily_returns[-1] - np.mean(daily_returns)) / np.std(daily_returns) if len(daily_returns) > 1 else 0
    features.append(z_score_returns)

    # Feature 4: Standard Deviation of Closing Prices (Volatility)
    price_volatility = np.std(closing_prices)
    features.append(price_volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(features) if np.std(features) != 0 else 1  # Avoid division by zero
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative reward for BUY-aligned features
        reward += 10 * features[1]  # Mild positive for selling if ATR indicates volatility

    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += 10 * features[0]  # Positive reward for momentum (EMA)
        elif trend_direction < -0.3:  # Downtrend
            reward += 10 * features[1]  # Positive reward for downward momentum (ATR)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Assuming Z-score < -1 indicates oversold
            reward += 15  # Buy signal
        elif features[2] > 1:  # Assuming Z-score > 1 indicates overbought
            reward += 15  # Sell signal
        else:
            reward -= 10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds