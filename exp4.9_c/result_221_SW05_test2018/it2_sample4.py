import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state.
    
    s: 120d raw state
    Returns ONLY new features (NOT including s or regime).
    """
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) for trend detection
    ema = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    
    # Feature 2: Price momentum (current closing price vs. 10-day EMA)
    price_momentum = (closing_prices[-1] - ema) / (ema if ema != 0 else 1)

    # Feature 3: Average True Range (ATR) for volatility measure
    high_prices = s[2::6]
    low_prices = s[3::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-period ATR

    # Feature 4: Price Relative to Bollinger Bands
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        distance_to_upper = (closing_prices[-1] - upper_band) / (upper_band if upper_band != 0 else 1)
        distance_to_lower = (closing_prices[-1] - lower_band) / (lower_band if lower_band != 0 else 1)
    else:
        distance_to_upper = 0
        distance_to_lower = 0

    # Combine features
    features = [price_momentum, atr, distance_to_upper, distance_to_lower]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract computed features
    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_std = np.std(features)  # Use historical std of features for dynamic thresholds
    low_risk_threshold = 0.4 * historical_std
    high_risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Expectation of upward momentum
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        elif features[0] < 0:
            reward += np.random.uniform(5, 10)

    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        if features[0] > 0:
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < low_risk_threshold:
        if trend_direction > trend_threshold and features[0] > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -trend_threshold and features[0] < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[2] > 0:  # Overbought condition
            reward += np.random.uniform(10, 20)
        elif features[3] < 0:  # Oversold condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]