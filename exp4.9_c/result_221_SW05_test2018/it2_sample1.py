import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state.

    s: 120d raw state
    Returns ONLY new features (NOT including s or regime).
    """
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices

    # Feature 1: Price momentum (current closing price vs. moving average of last 10 days)
    moving_average_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average_10) / (moving_average_10 if moving_average_10 != 0 else 1)

    # Feature 2: Average True Range (ATR) for volatility measure
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 3: Current price relative to Bollinger Bands (width)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        distance_to_upper = (closing_prices[-1] - upper_band) / (upper_band if upper_band != 0 else 1)
        distance_to_lower = (closing_prices[-1] - lower_band) / (lower_band if lower_band != 0 else 1)
    else:
        distance_to_upper = distance_to_lower = 0

    # Feature 4: Rate of Change (momentum) over the last 14 days
    if len(closing_prices) >= 15:
        rate_of_change = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15]
    else:
        rate_of_change = 0

    # Combine features into a single array
    features = [price_momentum, atr, distance_to_upper, distance_to_lower, rate_of_change]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract computed features
    price_momentum = features[0]
    atr = features[1]
    distance_to_upper = features[2]
    distance_to_lower = features[3]
    rate_of_change = features[4]

    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_std = np.std([risk_level])  # Use historical std for dynamic thresholds
    low_risk_threshold = 0.4 * historical_std
    high_risk_threshold = 0.7 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if price_momentum > 0:  # Expectation of upward momentum
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        else:
            reward += np.random.uniform(5, 10)

    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        if price_momentum > 0:  # Expectation of upward momentum
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3 and price_momentum > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3 and price_momentum < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if distance_to_upper > 0:  # Overbought condition
            reward += np.random.uniform(10, 20)
        elif distance_to_lower < 0:  # Oversold condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]