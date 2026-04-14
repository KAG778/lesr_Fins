import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]         # Extracting volumes

    # Feature 1: Crisis Indicator (standard deviation of the last 20 days)
    crisis_indicator = np.std(closing_prices[-20:])  # Volatility as a crisis measure

    # Feature 2: Trend Strength (using the difference from the 20-day moving average)
    moving_average_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    trend_strength = closing_prices[-1] - moving_average_20  # Current price - 20-day MA

    # Feature 3: Mean Reversion Indicator (distance from recent high/low)
    recent_high = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    recent_low = np.min(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    mean_reversion_distance = (closing_prices[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0

    features = [crisis_indicator, trend_strength, mean_reversion_distance]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    
    # Initialize reward
    reward = 0.0
    
    # Calculate relative thresholds based on historical std dev
    historical_std = np.std(enhanced_s[0:120])  # Standard deviation of historical prices
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -40 if features[1] > 0 else 5  # Strong negative for bullish features, mild positive for bearish features
    elif risk_level > risk_threshold_medium:
        reward += -20 if features[1] > 0 else 0  # Moderate negative for bullish features

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3 and features[1] > 0:  # Positive reward for bullish signals
            reward += 10
        elif trend_direction < -0.3 and features[1] < 0:  # Positive reward for bearish signals
            reward += 10

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0.3:  # Assuming feature[2] indicates oversold
            reward += 10  # Reward for buying in mean-reversion condition
        elif features[2] > 0.7:  # Assuming feature[2] indicates overbought
            reward += 10  # Reward for selling in mean-reversion condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds