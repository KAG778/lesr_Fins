import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes
    
    # 1. Bollinger Bands
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        bollinger_band_width = (upper_band - lower_band) / rolling_mean  # relative width
    else:
        bollinger_band_width = 0.0  # Not enough data

    # 2. MACD (12-day EMA - 26-day EMA)
    if len(closing_prices) >= 26:
        short_ema = np.mean(closing_prices[-12:])  # Simple approximation of EMA
        long_ema = np.mean(closing_prices[-26:])   # Simple approximation of EMA
        macd = short_ema - long_ema
    else:
        macd = 0.0  # Not enough data

    # 3. Volume Weighted Average Price (VWAP)
    if len(trading_volumes) > 0:
        vwap = np.sum(closing_prices * trading_volumes) / np.sum(trading_volumes) if np.sum(trading_volumes) > 0 else 0.0
    else:
        vwap = 0.0

    features = [
        bollinger_band_width,  # Feature 1: Bollinger Bands Width
        macd,                  # Feature 2: MACD
        vwap                   # Feature 3: VWAP
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds based on standard deviation (example values)
    risk_threshold_high = np.mean(risk_level) + 2 * np.std(risk_level)
    risk_threshold_medium = np.mean(risk_level) + np.std(risk_level)

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(5, 10)    # Mild positive for SELL
    elif risk_level > risk_threshold_medium:
        reward -= np.random.uniform(10, 20)    # Moderate negative for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for upward momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downward momentum

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward for taking advantage of mean reversion

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds