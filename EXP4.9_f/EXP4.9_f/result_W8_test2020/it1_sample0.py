import numpy as np

def revise_state(s):
    # Extract relevant price data
    closing_prices = s[0::6]  # Closing prices
    high_prices = s[2::6]     # High prices
    low_prices = s[3::6]      # Low prices
    volume = s[4::6]          # Trading volume

    # Feature 1: Price Momentum (last 5 days)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0

    # Feature 2: Average True Range (ATR)
    true_ranges = high_prices - low_prices
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Feature 3: Bollinger Bands (Upper and Lower Bands)
    mean_price = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    upper_band = mean_price + (2 * std_dev)  # Upper Bollinger Band
    lower_band = mean_price - (2 * std_dev)  # Lower Bollinger Band
    price_position = (closing_prices[-1] - lower_band) / (upper_band - lower_band) if upper_band - lower_band != 0 else 0  # Normalized position within bands

    # Feature 4: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices[-20:] * volume[-20:]) / np.sum(volume[-20:]) if np.sum(volume[-20:]) != 0 else 0

    features = [price_momentum, atr, price_position, vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk levels
    risk_threshold_high = 0.7  # Placeholder for high risk threshold
    risk_threshold_moderate = 0.4  # Placeholder for moderate risk threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY signals
    elif risk_level > risk_threshold_moderate:
        reward += 20  # Mildly positive reward for SELL signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Reward for positive momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Reward for negative momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features (sideways market)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within the specified bounds