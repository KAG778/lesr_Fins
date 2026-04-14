import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state.
    
    s: 120d raw state
    Returns ONLY new features (NOT including s or regime).
    """
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes
    
    # Feature 1: Price momentum - current closing price vs. moving average of last 10 days
    moving_average_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average_10) / (moving_average_10 if moving_average_10 != 0 else 1)

    # Feature 2: Average True Range (ATR) for volatility measure
    true_ranges = np.maximum(high_prices[-5:] - low_prices[-5:], np.abs(closing_prices[-5:] - closing_prices[-6:-1]))
    atr = np.mean(true_ranges) if len(true_ranges) > 0 else 0

    # Feature 3: Bollinger Bands - width
    if len(closing_prices) >= 20:
        moving_average_20 = np.mean(closing_prices[-20:])
        std_dev_20 = np.std(closing_prices[-20:])
        bollinger_upper = moving_average_20 + (2 * std_dev_20)
        bollinger_lower = moving_average_20 - (2 * std_dev_20)
        bollinger_width = (bollinger_upper - bollinger_lower) / moving_average_20 if moving_average_20 != 0 else 0
    else:
        bollinger_width = 0

    # Feature 4: Current price relative to the 20-day moving average
    ma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    price_relative_to_ma = (closing_prices[-1] - ma_20) / (ma_20 if ma_20 != 0 else 1)

    return np.array([price_momentum, atr, bollinger_width, price_relative_to_ma])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    price_momentum = features[0]
    atr = features[1]
    bollinger_width = features[2]
    price_relative_to_ma = features[3]

    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_std = np.std([risk_level])  # Use historical std of risk levels for dynamic thresholds
    low_risk_threshold = 0.4 * historical_std
    high_risk_threshold = 0.7 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if price_momentum > 0:  # Price momentum suggests BUY
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        elif price_momentum < 0:  # Price momentum suggests SELL
            reward += np.random.uniform(5, 10)

    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        if price_momentum > 0:
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3 and price_momentum > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3 and price_momentum < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if price_relative_to_ma < -0.1:  # Oversold condition
            reward += np.random.uniform(10, 20)
        elif price_relative_to_ma > 0.1:  # Overbought condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]