import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Average True Range (ATR) over the last 14 days
    def calculate_atr(prices, period=14):
        high = prices[::6]  # Assuming high prices are at every 6th index
        low = prices[1::6]   # Assuming low prices are at every 6th index + 1
        close = prices[2::6] # Assuming close prices are at every 6th index + 2
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        return np.mean(tr[-period:]) if len(tr) >= period else np.nan

    atr = calculate_atr(closing_prices)

    # Feature 2: Bollinger Bands (20-day moving average and standard deviation)
    def calculate_bollinger_bands(prices, period=20):
        if len(prices) >= period:
            rolling_mean = np.mean(prices[-period:])
            rolling_std = np.std(prices[-period:])
            upper_band = rolling_mean + (rolling_std * 2)
            lower_band = rolling_mean - (rolling_std * 2)
            return upper_band, lower_band
        else:
            return np.nan, np.nan

    upper_band, lower_band = calculate_bollinger_bands(closing_prices)

    # Feature 3: Exponential Moving Average (EMA) over the last 14 days
    def calculate_ema(prices, period=14):
        if len(prices) < period:
            return np.nan
        return np.mean(prices[-period:])  # Simple EMA for demonstration purposes

    ema = calculate_ema(closing_prices)

    features = [atr, upper_band, lower_band, ema]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk levels
    risk_threshold_high = 0.7  # Define high risk threshold based on historical data
    risk_threshold_moderate = 0.4  # Define moderate risk threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 10)   # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(5, 15)    # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        features = enhanced_s[123:]
        # Assuming features[1] (upper_band) and features[2] (lower_band) carry significant information
        if features[2] < enhanced_s[0] < features[1]:  # Current price is between the bands
            reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]