import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Average True Range (ATR) over the last 14 days
    def calculate_atr(prices, high_prices, low_prices, period=14):
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - prices[:-1]), 
                                   np.abs(low_prices[1:] - prices[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else 0
        return atr

    closing_prices = s[0:120:6]
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    atr = calculate_atr(closing_prices, high_prices, low_prices)
    features.append(atr)
    
    # Feature 2: Correlation between price change and volume
    price_changes = np.diff(closing_prices)
    volume_changes = np.diff(s[4:120:6])
    if len(price_changes) > 0 and len(volume_changes) > 0:
        correlation = np.corrcoef(price_changes[-10:], volume_changes[-10:])[0, 1]
    else:
        correlation = 0
    features.append(correlation)

    # Feature 3: Exponential Moving Average (EMA) of closing prices
    def calculate_ema(prices, period=10):
        if len(prices) < period:
            return 0
        ema = np.zeros_like(prices)
        k = 2 / (period + 1)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * k + ema[i - 1]
        return ema[-1]

    ema = calculate_ema(closing_prices)
    features.append(ema)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(enhanced_s[123:])  # Based on features
    high_risk_threshold = 0.7 * historical_volatility
    medium_risk_threshold = 0.4 * historical_volatility

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward += -50  # Strong negative reward for BUY-aligned features
        reward += 10    # Mild positive reward for SELL-aligned features
    elif risk_level > medium_risk_threshold:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        if trend_direction > 0:
            reward += 20  # Reward for upward trend
        else:
            reward += 20  # Reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)