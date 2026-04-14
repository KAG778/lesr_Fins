import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes

    # Feature 1: Adaptive Moving Average (10 days) adjusted by volatility
    if len(closing_prices) > 9:
        moving_avg = np.mean(closing_prices[-10:])
        adjusted_ma = moving_avg + np.std(closing_prices[-10:])  # Adjusted MA based on volatility
    else:
        adjusted_ma = 0
    
    # Feature 2: Volatility-Adjusted RSI Calculation
    def calculate_volatility_adjusted_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        if loss == 0:
            return 100
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_value = calculate_volatility_adjusted_rsi(closing_prices[-14:])
    
    # Feature 3: Average True Range (ATR) for volatility measurement
    true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                              np.maximum(closing_prices[1:] - closing_prices[:-1], 
                                         closing_prices[:-1] - closing_prices[1:]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) > 13 else 0

    features = [adjusted_ma, rsi_value, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk and trend
    historical_std = np.std(enhanced_s[123:])  # Assuming features start from index 123
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY
        reward += 10  # MILD POSITIVE for SELL
    elif risk_level > risk_threshold_moderate:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_moderate:
        features = enhanced_s[123:]
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += 15
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += 15

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        features = enhanced_s[123:]
        if features[1] < 30:  # Oversold condition
            reward += 10
        elif features[1] > 70:  # Overbought condition
            reward += 10

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds