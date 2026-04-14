import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) for trend detection
    def compute_ema(prices, span=14):
        if len(prices) < span:
            return np.nan
        weights = np.exp(np.linspace(-1., 0., span))
        weighted_prices = weights / weights.sum()
        ema = np.convolve(prices, weighted_prices, mode='valid')[-1]
        return ema
    
    ema = compute_ema(closing_prices)

    # Feature 2: Average True Range (ATR) for volatility detection
    def compute_atr(prices, n=14):
        if len(prices) < n + 1:
            return np.nan
        high = prices[::6][1:]  # High prices
        low = prices[::6][1:]   # Low prices
        close = prices[::6][:-1]  # Closing prices
        tr = np.maximum(high - low, np.maximum(abs(high - close), abs(low - close)))
        return np.mean(tr[-n:])

    atr = compute_atr(s)

    # Feature 3: Standard deviation of returns for crisis detection
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns)

    features.append(ema)
    features.append(atr)
    features.append(volatility)

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
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std
    low_volatility_threshold = 0.3 * historical_std
    high_volatility_threshold = 0.6 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -50  # STRONG NEGATIVE reward for BUY-aligned features
        reward += 10 * features[1]  # Mild positive for SELL based on ATR
    elif risk_level > medium_risk_threshold:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < medium_risk_threshold:
        if trend_direction > 0.3:
            reward += 10 * features[0]  # Positive reward for upward trend (EMA)
        elif trend_direction < -0.3:
            reward += 10 * features[1]  # Positive reward for downward trend (ATR)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < low_volatility_threshold and risk_level < low_volatility_threshold:
        if features[2] < low_volatility_threshold:  # Assuming low volatility indicates mean-reversion
            reward += 10  # Reward for mean-reversion logic

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > high_volatility_threshold and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds