import numpy as np

def revise_state(s):
    closing_prices = s[0::6]
    volumes = s[4::6]

    # Feature 1: Price Momentum (percentage change from the last day)
    if len(closing_prices) >= 2:
        price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    else:
        price_momentum = 0.0

    # Feature 2: Historical Volatility (standard deviation of closing prices over the last 20 days)
    historical_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0

    # Feature 3: Market Correlation (dummy implementation, replace with actual market data)
    # Assuming we have a function to get market closing prices
    market_prices = np.random.normal(loc=np.mean(closing_prices), scale=np.std(closing_prices), size=len(closing_prices))  # Placeholder for actual market data
    market_correlation = np.corrcoef(closing_prices[-20:], market_prices[-20:])[0, 1] if len(closing_prices) >= 20 else 0.0

    # Feature 4: Drawdown Duration (average drawdown duration in the last 20 days)
    drawdowns = []
    peak = closing_prices[0]
    for price in closing_prices[1:]:
        if price < peak:
            drawdown = peak - price
            drawdowns.append(drawdown)
        else:
            peak = price
    avg_drawdown_duration = np.mean(drawdowns) if drawdowns else 0.0

    # Return the new features as a numpy array
    features = [price_momentum, historical_volatility, market_correlation, avg_drawdown_duration]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Define dynamic thresholds based on historical data
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    trend_threshold = 0.3

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        if features[0] > 0:  # Positive momentum (BUY signal)
            reward = -50  # Strong negative reward for BUY
        else:  # Negative momentum (SELL signal)
            reward = 10  # Mild positive reward for SELL
    elif risk_level > risk_threshold_medium:
        if features[0] > 0:  # Positive momentum
            reward = -20  # Moderate negative reward for BUY

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > trend_threshold and features[0] > 0:  # Positive momentum in uptrend
            reward = 20  # Positive reward
        elif trend_direction < -trend_threshold and features[0] < 0:  # Negative momentum in downtrend
            reward = 20  # Positive reward

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if features[2] < 0:  # Assuming negative correlation indicates sell signal
            reward = 15  # Reward for mean-reversion
        else:
            reward = -5  # Penalty for chasing trends

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]