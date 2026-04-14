import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    trading_volumes = s[4:120:6]  # Extract trading volumes
    
    # Feature 1: 14-day Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 2: 14-day moving average
    moving_average = np.mean(closing_prices[-14:]) if len(closing_prices) >= 14 else 0
    
    # Feature 3: Price volatility (standard deviation of returns over the last 14 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-14:]) if len(returns) >= 14 else 0
    
    # Feature 4: Average trading volume over the last 14 days
    avg_volume = np.mean(trading_volumes[-14:]) if len(trading_volumes) >= 14 else 0

    features = [rsi, moving_average, volatility, avg_volume]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Using the features to assess standard deviation
    high_risk_threshold = historical_std
    low_volatility_threshold = 0.2 * historical_std  # Example threshold for low volatility

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:  # Adjusting threshold for high risk dynamically
        reward -= 50  # Strong negative reward for risking buying
        reward += 10 * (1 - np.clip(enhanced_s[123][0], 0, 1))  # Mild positive for selling
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative for risky buying

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_volatility_threshold:  # Low risk for trend following
        if trend_direction > 0:  # Uptrend
            reward += 30  # Positive reward for aligning with upward trend
        elif trend_direction < 0:  # Downtrend
            reward += 30  # Positive reward for aligning with downward trend
    
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:  # Low risk during sideways movement
        if enhanced_s[123][0] < 30:  # Assuming RSI < 30 indicates oversold
            reward += 20  # Reward for buying in oversold conditions
        elif enhanced_s[123][0] > 70:  # Assuming RSI > 70 indicates overbought
            reward += 20  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_volatility_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds