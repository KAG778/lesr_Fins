import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Moving Average Convergence Divergence (MACD)
    def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
        if len(prices) < long_window:
            return 0, 0  # Not enough data
        short_ema = np.mean(prices[-short_window:])
        long_ema = np.mean(prices[-long_window:])
        macd = short_ema - long_ema
        signal = np.mean(prices[-signal_window:])
        return macd - signal  # MACD - Signal Line

    macd_signal = calculate_macd(closing_prices)
    features.append(macd_signal)

    # Feature 2: Average True Range (ATR) for volatility
    def calculate_atr(prices, high, low, period=14):
        if len(prices) < period:
            return 0  # Not enough data
        tr = np.maximum(high[-period:] - low[-period:], 
                        np.maximum(np.abs(high[-period:] - prices[-period-1:-1]),
                                   np.abs(low[-period:] - prices[-period-1:-1])))
        return np.mean(tr)

    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    atr = calculate_atr(closing_prices, high_prices, low_prices)
    features.append(atr)

    # Feature 3: Cumulative Volume Change
    if len(volumes) >= 5:
        cumulative_volume_change = np.sum(volumes[-5:]) - np.sum(volumes[-10:-5])
    else:
        cumulative_volume_change = 0
    features.append(cumulative_volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate relative thresholds based on historical data
    historical_std = np.std(enhanced_s[0:120])  # Use the raw state for variability
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for BUY
        reward += 10 if enhanced_s[123] < 0 else 0  # Mild positive for SELL
    elif risk_level > 0.4 * historical_std:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold:
            reward += 30  # Strong positive for upward features
        elif trend_direction < -trend_threshold:
            reward += 30  # Strong positive for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure reward is within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward