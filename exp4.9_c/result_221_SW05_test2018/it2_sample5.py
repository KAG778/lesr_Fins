import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes
    
    # Feature 1: Price momentum (current closing price vs. moving average of last 10 days)
    moving_average = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average) / (moving_average if moving_average != 0 else 1)

    # Feature 2: Average True Range (ATR) for volatility measure
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR

    # Feature 3: Bollinger Bands (current price relative to upper and lower bands)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
    else:
        upper_band = lower_band = closing_prices[-1]  # Fallback to last price
    distance_to_upper = (closing_prices[-1] - upper_band) / (upper_band if upper_band != 0 else 1)
    distance_to_lower = (closing_prices[-1] - lower_band) / (lower_band if lower_band != 0 else 1)

    # Feature 4: Rate of Change (Momentum)
    if len(closing_prices) >= 14:
        roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15]  # 14-period ROC
    else:
        roc = 0

    features = [price_momentum, atr, distance_to_upper, distance_to_lower, roc]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    price_momentum = features[0]
    atr = features[1]
    distance_to_upper = features[2]
    distance_to_lower = features[3]
    roc = features[4]

    reward = 0.0
    
    # Calculate historical thresholds for risk management
    historical_std = np.std([risk_level])  # Use historical std of risk
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
        if distance_to_upper > 0:  # Overbought condition
            reward += np.random.uniform(10, 20)
        elif distance_to_lower < 0:  # Oversold condition
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]