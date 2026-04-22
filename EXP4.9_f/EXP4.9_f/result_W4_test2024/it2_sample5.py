import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Closing prices for 20 days
    high_prices = s[2:120:6]     # High prices for 20 days
    low_prices = s[3:120:6]      # Low prices for 20 days
    volumes = s[4:120:6]         # Trading volumes for 20 days
    days = len(closing_prices)

    # Feature 1: Price Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if days >= 5 else 0

    # Feature 2: Average True Range (ATR) for Volatility
    def average_true_range(high, low, close, window):
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        return np.mean(tr[-window:]) if len(tr) >= window else 0
    
    atr_value = average_true_range(high_prices, low_prices, closing_prices, 14)  # 14-day ATR

    # Feature 3: Price Relative to 200-day Moving Average
    if days >= 200:
        ma200 = np.mean(closing_prices[-200:])
        price_relative_to_ma200 = (closing_prices[-1] - ma200) / ma200
    else:
        price_relative_to_ma200 = 0

    # Feature 4: Volume Oscillator (difference of short and long term moving averages of volume)
    short_volume_ma = np.mean(volumes[-5:]) if days >= 5 else 0  # Short-term volume average
    long_volume_ma = np.mean(volumes[-20:]) if days >= 20 else 0  # Long-term volume average
    volume_oscillator = short_volume_ma - long_volume_ma

    features = [roc, atr_value, price_relative_to_ma200, volume_oscillator]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Use features to calculate a standard deviation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # STRONG NEGATIVE for BUY-aligned features
        if trend_direction < 0:  # Mild positive for SELL-aligned features
            reward += 10
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Positive reward for trend-following

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]