import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Get closing prices (20 days)
    volumes = s[4:120:6]          # Get volumes (20 days)
    features = []

    # 1. Average True Range (ATR) for volatility
    def compute_atr(prices, period=14):
        tr = np.zeros(len(prices)-1)
        for i in range(1, len(prices)):
            tr[i-1] = max(prices[i] - prices[i-1], 
                           abs(prices[i] - prices[i-1]), 
                           abs(prices[i-1] - prices[i]))
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    atr = compute_atr(closing_prices)
    features.append(atr)

    # 2. Momentum Oscillator (Rate of Change)
    if len(closing_prices) >= 2:
        momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        momentum = 0
    features.append(momentum)

    # 3. Moving Average Convergence Divergence (MACD)
    def compute_macd(prices, short_window=12, long_window=26):
        short_ema = np.mean(prices[-short_window:]) if len(prices) >= short_window else prices[-1]
        long_ema = np.mean(prices[-long_window:]) if len(prices) >= long_window else prices[-1]
        return short_ema - long_ema

    macd = compute_macd(closing_prices)
    features.append(macd)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical standard deviation for relative thresholds
    historical_std = np.std(enhanced_s[123:])  # Use feature array for volatility

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE for BUY-aligned features
        reward += +10 if trend_direction < 0 else -5  # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 20 * (1 - abs(trend_direction))  # Scale reward based on trend strength
        else:  # Downtrend
            reward += 20 * (1 - abs(trend_direction))  # Scale reward based on trend strength

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for potential mean-reversion features
        reward -= 5   # Penalize for breakout-chasing features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > historical_std * 1.5:  # Example condition for high volatility
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds