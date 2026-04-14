import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[1::6]      # Extract high prices
    low_prices = s[2::6]       # Extract low prices
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

    # Feature 2: 14-Day Average True Range (ATR)
    def compute_atr(highs, lows, closes, n=14):
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        return np.mean(tr[-n:]) if len(tr) >= n else np.nan

    atr = compute_atr(high_prices, low_prices, closing_prices)
    features.append(atr if not np.isnan(atr) else 0)

    # Feature 3: Z-score of Daily Returns (to detect volatility)
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

    # Calculate historical volatility for dynamic thresholds
    historical_std = np.std(features) if len(features) > 0 else 1
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative reward for BUY signals
        # Allow mild positive for selling (if ATR indicates potential for downward movement)
        reward += 10 * features[1]  # ATR-based selling signal
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += 10 * features[0]  # Positive reward for momentum (EMA)
        elif trend_direction < -0.3:  # Downtrend
            reward += 10 * features[1]  # Positive reward for volatility (ATR)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < -1:  # Assuming z-score < -1 indicates oversold
            reward += 10  # Buy signal
        elif features[2] > 1:  # Assuming z-score > 1 indicates overbought
            reward += 10  # Sell signal

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds