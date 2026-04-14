import numpy as np

def revise_state(s):
    features = []

    # Closing prices for the last 20 days
    closing_prices = s[0:120:6]  # every 6th element is a closing price
    volumes = s[4:120:6]          # extract trading volumes

    # Feature 1: Volatility-Adjusted Momentum
    price_change = closing_prices[-1] - closing_prices[-6]  # Momentum over the last 5 days
    volatility = np.std(closing_prices[-5:])  # Standard deviation of the last 5 days
    volatility_adjusted_momentum = price_change / (volatility + 1e-5)  # Avoiding division by zero
    features.append(volatility_adjusted_momentum)

    # Feature 2: Bollinger Bands (20-day SMA and std deviation)
    if len(closing_prices) >= 20:
        sma_20 = np.mean(closing_prices[-20:])
        std_20 = np.std(closing_prices[-20:])
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        price_position = (closing_prices[-1] - lower_band) / (upper_band - lower_band)  # Normalized position
    else:
        price_position = 0.5  # Neutral position when not enough data
    features.append(price_position)

    # Feature 3: Market Breadth Indicator (Advance-Decline Line)
    if len(volumes) >= 20:
        avg_volume = np.mean(volumes[-20:])
        breadth_indicator = (volumes[-1] - avg_volume) / (avg_volume + 1e-5)  # Normalize against average volume
    else:
        breadth_indicator = 0.0  # Neutral when not enough data
    features.append(breadth_indicator)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical data
    historical_risk_threshold = 0.5  # Example threshold based on historical performance
    historical_trend_threshold = 0.3  # Example threshold based on historical performance

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk_threshold:
        reward += -40  # Strong negative for BUY when risk is high
        reward += 10 * np.random.random()  # Mild positive for SELL
        return np.clip(reward, -100, 100)  # Early exit

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > historical_trend_threshold and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20  # Positive reward for bullish alignment
        else:
            reward += 20  # Positive reward for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < historical_trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensuring reward stays within bounds