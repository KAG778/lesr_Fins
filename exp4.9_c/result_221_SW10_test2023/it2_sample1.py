import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    num_prices = len(closing_prices)

    # Feature 1: Volatility Adjusted Momentum
    if num_prices > 5:
        momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]
        if len(daily_returns) > 0:
            volatility = np.std(daily_returns)
            vol_adjusted_momentum = momentum / (volatility + 1e-9)  # Avoid division by zero
        else:
            vol_adjusted_momentum = momentum
    else:
        vol_adjusted_momentum = 0
    features.append(vol_adjusted_momentum)

    # Feature 2: Z-score of Returns
    if num_prices > 1:
        daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        z_score = (daily_returns[-1] - mean_return) / (std_return + 1e-9)  # Normalize
    else:
        z_score = 0
    features.append(z_score)

    # Feature 3: Bollinger Bands %B
    if num_prices >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        current_price = closing_prices[-1]
        if upper_band - lower_band > 0:  # Avoid division by zero
            bollinger_percent_b = (current_price - lower_band) / (upper_band - lower_band)
        else:
            bollinger_percent_b = 0
    else:
        bollinger_percent_b = 0
    features.append(bollinger_percent_b)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(features)
    high_risk_threshold = historical_std * 1.5  # Using 1.5 standard deviations
    low_risk_threshold = historical_std * 0.5   # Using 0.5 standard deviations

    # Priority 1: Risk Management
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # Positive momentum indicates a BUY signal
            reward -= np.random.uniform(40, 60)  # Strong negative for BUY
        else:
            reward += np.random.uniform(5, 15)  # Mild positive for SELL
    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Positive momentum
            reward -= np.random.uniform(20, 30)

    # Priority 2: Trend Following (only when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(10, 20)  # Reward for alignment
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(10, 20)  # Reward for alignment

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < -1:  # Z-score indicating extreme oversold condition
            reward += np.random.uniform(5, 15)  # Reward for buying in oversold conditions
        elif features[1] > 1:  # Z-score indicating extreme overbought condition
            reward -= np.random.uniform(5, 15)  # Penalty for selling in overbought conditions

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return float(np.clip(reward, -100, 100))