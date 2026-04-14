import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state.
    
    s: 120d raw state
    Returns ONLY new features (NOT including s or regime).
    """
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes
    
    # Feature 1: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0

        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(closing_prices)

    # Feature 2: Bollinger Bands
    def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
        if len(prices) < window:
            return np.nan, np.nan  # Not enough data

        rolling_mean = np.mean(prices[-window:])
        rolling_std = np.std(prices[-window:])
        upper_band = rolling_mean + (num_std_dev * rolling_std)
        lower_band = rolling_mean - (num_std_dev * rolling_std)
        return upper_band, lower_band

    upper_band, lower_band = calculate_bollinger_bands(closing_prices)

    # Feature 3: Average True Range (ATR)
    def calculate_atr(prices, high_prices, low_prices, period=14):
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - prices[:-1]), 
                                   np.abs(low_prices[1:] - prices[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else 0
        return atr

    high_prices = s[2::6]
    low_prices = s[3::6]
    atr = calculate_atr(closing_prices, high_prices, low_prices)

    features = [rsi, upper_band, lower_band, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0
    
    # Thresholds based on historical standard deviations
    risk_threshold_high = 0.7  # This can be adjusted based on historical data
    trend_threshold = 0.3       # Absolute trend threshold for significant trend
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50) if features[0] > 70 else np.random.uniform(0, 10)  # RSI condition
        # Mild positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10) if features[0] < 30 else 0  # RSI condition

    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20) if features[0] > 70 else 0  # RSI condition

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold and features[0] > 50:  # Bullish momentum
            reward += np.random.uniform(10, 20)  # Align with upward trend
        elif trend_direction < -trend_threshold and features[0] < 50:  # Bearish momentum
            reward += np.random.uniform(10, 20)  # Align with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[0] < 30:  # Oversold condition
            reward += np.random.uniform(10, 20)
        elif features[0] > 70:  # Overbought condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]