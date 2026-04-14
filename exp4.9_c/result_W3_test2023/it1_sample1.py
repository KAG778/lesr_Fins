import numpy as np

def revise_state(s):
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    features = []

    # Feature 1: 14-day Average True Range (ATR) for volatility measurement
    def calculate_atr(prices, period=14):
        high_low = np.array(prices[2::6]) - np.array(prices[3::6])  # High - Low
        high_prev_close = np.abs(np.array(prices[2::6]) - np.array(prices[1::6]))  # High - Previous Close
        low_prev_close = np.abs(np.array(prices[3::6]) - np.array(prices[1::6]))  # Low - Previous Close
        tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
        return np.mean(tr[-period:]) if len(tr) >= period else 0

    atr = calculate_atr(s)

    # Feature 2: Z-Score of the last closing price based on historical data
    historical_mean = np.mean(closing_prices)
    historical_std = np.std(closing_prices)
    z_score = (closing_prices[-1] - historical_mean) / historical_std if historical_std != 0 else 0

    # Feature 3: 5-day Momentum (current - past)
    momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 6 else 0

    features = [atr, z_score, momentum]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(enhanced_s[123:])  # Using features for volatility
    risk_threshold_high = 0.7 * historical_volatility
    risk_threshold_moderate = 0.4 * historical_volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -np.random.uniform(30, 50)  # Strong negative for BUY-aligned
        reward += np.random.uniform(5, 10)     # Mild positive for SELL-aligned
    elif risk_level > risk_threshold_moderate:
        reward += -10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Positive reward for upward momentum
        else:  # Downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Oversold condition
            reward += 15  # Reward for buy
        else:  # Overbought condition
            reward += -15  # Penalize for sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]