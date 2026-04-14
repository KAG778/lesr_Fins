import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract volumes

    # Feature 1: Price momentum (simple return over the last 10 days)
    price_momentum = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if closing_prices[-11] != 0 else 0

    # Feature 2: Average trading volume over the last 10 days
    avg_volume = np.mean(volumes[-10:]) if len(volumes[-10:]) > 0 else 0

    # Feature 3: Bollinger Bands for last 20 days
    moving_avg = np.mean(closing_prices[-20:])
    moving_std = np.std(closing_prices[-20:])
    lower_band = moving_avg - (2 * moving_std)
    upper_band = moving_avg + (2 * moving_std)

    # Feature indicating if the current closing price is above or below the bands
    price_above_upper_band = 1.0 if closing_prices[-1] > upper_band else 0.0
    price_below_lower_band = 1.0 if closing_prices[-1] < lower_band else 0.0

    # Feature 4: Average daily return over the last 20 days
    daily_returns = np.diff(closing_prices[-20:]) / closing_prices[-21:-1]
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0

    # Combine features into a single array
    features = [price_momentum, avg_volume, price_above_upper_band, price_below_lower_band, avg_daily_return]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate thresholds based on historical data
    avg_risk_level = 0.5  # Hypothetical average risk level (to be replaced with historical data)
    std_risk_level = 0.2  # Hypothetical standard deviation of risk level (replace with actual)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > avg_risk_level + 1.5 * std_risk_level:  # Strong risk
        reward -= 50  # Strong negative reward for risky BUY-aligned features
        return reward  # Immediate return to prioritize risk management
    elif risk_level > avg_risk_level + 0.5 * std_risk_level:  # Moderate risk
        reward -= 20  # Moderate negative reward for BUY signals

    # Extract features
    features = enhanced_s[123:]

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < avg_risk_level:
        if trend_direction > 0 and features[0] > 0:  # Upward momentum
            reward += 30  # Strong positive reward for upward alignment
        elif trend_direction < 0 and features[3] > 0:  # Downward momentum
            reward += 30  # Strong positive reward for downward alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < avg_risk_level:
        if features[2] > 0:  # Price above upper band
            reward -= 10  # Penalize for chasing breakouts
        elif features[3] > 0:  # Price below lower band
            reward += 15  # Reward for buying in an oversold market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < avg_risk_level:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))
    
    return reward