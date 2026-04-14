import numpy as np

def revise_state(s):
    num_days = 20
    closing_prices = s[0:num_days*6:6]  # Extract closing prices
    volumes = s[4:num_days*6:6]          # Extract volumes

    # Calculate RSI
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0).mean()
        loss = -np.where(delta < 0, delta, 0).mean()  # Use negative losses
        rs = gain / loss if loss > 0 else 0  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices[-14:])  # Using the last 14 days for RSI
    
    # Calculate moving averages
    sma_10 = np.mean(closing_prices[-10:])  # Simple moving average of the last 10 days
    sma_5 = np.mean(closing_prices[-5:])    # Simple moving average of the last 5 days

    # Calculate volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    volatility = np.std(returns)

    # Prepare features array
    features = [rsi, sma_10, sma_5, volatility]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical data
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += np.random.uniform(-50, -30)  # Strong negative reward for BUY-aligned features
    elif risk_level > risk_threshold_medium:
        reward += np.random.uniform(-20, -10)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 30:  # Oversold condition
            reward += 10  # Reward for mean-reversion buy
        elif enhanced_s[123] > 70:  # Overbought condition
            reward += 10  # Reward for mean-reversion sell

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)