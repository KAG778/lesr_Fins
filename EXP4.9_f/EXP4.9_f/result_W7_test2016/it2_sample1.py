import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Average True Range (ATR) over the last 14 days (measures volatility)
    def calculate_atr(prices, volumes, period=14):
        if len(prices) < period:
            return 0  # Not enough data
        
        tr = np.zeros(len(prices) - 1)
        for i in range(1, len(prices)):
            high_low = prices[i] - prices[i - 1]
            high_close = abs(prices[i] - prices[i - 1])
            low_close = abs(prices[i] - prices[i - 1])
            tr[i - 1] = max(high_low, high_close, low_close)

        return np.mean(tr[-period:])  # Average the true range over the period

    atr = calculate_atr(closing_prices, volumes)
    features.append(atr)

    # Feature 2: Exponential Moving Average (EMA) over the last 14 days
    def calculate_ema(prices, period=14):
        if len(prices) < period:
            return prices[-1]  # Fallback to the last closing price if not enough data
        
        multiplier = 2 / (period + 1)
        ema = prices[-period]  # Start with the first price in the series
        for price in prices[-period+1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    ema = calculate_ema(closing_prices)
    features.append(ema)

    # Feature 3: Rate of Change (ROC) over the last 14 days
    def calculate_roc(prices, period=14):
        if len(prices) < period:
            return 0  # Not enough data
        return (prices[-1] - prices[-period]) / prices[-period] if prices[-period] != 0 else 0

    roc = calculate_roc(closing_prices)
    features.append(roc)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[0:120])  # Use the raw state for variability
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10 if enhanced_s[123] < 0 else 0  # Mild positive for SELL-aligned features
    elif risk_level > 0.4 * historical_std:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold:
            reward += 30  # Strong positive for upward momentum
        elif trend_direction < -trend_threshold:
            reward += 30  # Strong positive for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 20  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)