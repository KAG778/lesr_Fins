import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Exponential Moving Average (EMA) Slope
    def ema_slope(prices, span=10):
        if len(prices) < span:
            return 0
        ema = np.zeros(len(prices))
        ema[0] = prices[0]
        alpha = 2 / (span + 1)
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1]
        return ema[-1] - ema[-2]  # Slope is the change between last two EMAs

    ema_slope_value = ema_slope(closing_prices)

    # Feature 2: Average True Range (ATR)
    def average_true_range(prices, period=14):
        if len(prices) < period:
            return 0
        high_low = np.diff(prices[::6][:, 2])  # High prices
        high_close = np.abs(prices[::6][:, 2] - prices[::6][:, 1])  # High - Previous Close
        low_close = np.abs(prices[::6][:, 1] - prices[::6][:, 3])  # Previous Close - Low
        true_ranges = np.maximum(high_low, np.maximum(high_close, low_close))
        return np.mean(true_ranges[-period:])

    atr_value = average_true_range(s.reshape(-1, 6), period=14)

    # Feature 3: Z-Score of RSI
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_value = calculate_rsi(closing_prices)
    # Assuming historical RSI data is available for z-score calculation
    rsi_mean = 50  # Hypothetical mean of historical RSI
    rsi_std = 30   # Hypothetical standard deviation of historical RSI
    z_score_rsi = (rsi_value - rsi_mean) / rsi_std if rsi_std != 0 else 0

    features = [ema_slope_value, atr_value, z_score_rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Determine thresholds based on historical std (hypothetical values)
    risk_threshold_high = 0.7
    risk_threshold_low = 0.4
    trend_threshold_high = 0.3
    trend_threshold_low = -0.3
    volatility_threshold = 0.6

    reward = 0.0
    
    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY
        reward += 10  # Mild positive for SELL
    elif risk_level > risk_threshold_low:
        reward -= 20  # Moderate negative for BUY

    # Priority 2: Trend Following
    elif abs(trend_direction) > trend_threshold_high and risk_level < risk_threshold_low:
        if trend_direction > 0:
            reward += 20  # Reward for positive trend
        else:
            reward += 20  # Reward for negative trend

    # Priority 3: Sideways / Mean Reversion
    elif trend_threshold_low < trend_direction < trend_threshold_high and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion in sideways markets

    # Priority 4: High Volatility
    if volatility_level > volatility_threshold and risk_level < risk_threshold_low:
        reward *= 0.5  # Reduce reward magnitude

    # Clamp the reward to be within [-100, 100]
    return np.clip(reward, -100, 100)