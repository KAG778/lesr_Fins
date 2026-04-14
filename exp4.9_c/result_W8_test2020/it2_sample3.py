import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    
    # 1. 20-day moving average for trend detection
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(moving_average)
    
    # 2. Price momentum (current price minus price from N days ago)
    N = 5
    price_momentum = closing_prices[-1] - closing_prices[-N] if len(closing_prices) >= N else 0.0
    features.append(price_momentum)
    
    # 3. Volatility over the last 20 days (standard deviation)
    volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(volatility)
    
    # 4. Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if len(closing_prices) >= 2 and closing_prices[-2] != 0 else 0.0
    features.append(roc)
    
    # 5. Bollinger Bands: Z-score for mean reversion signals
    rolling_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    z_score = (closing_prices[-1] - rolling_mean) / rolling_std if rolling_std != 0 else 0.0
    features.append(z_score)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Calculate dynamic thresholds based on historical data (std deviation of features)
    historical_std = np.std(enhanced_s[123:])  # Considering features for dynamic thresholds
    risk_high_threshold = 0.7 * historical_std
    risk_moderate_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1: Risk Management
    if risk_level > risk_high_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY actions
    elif risk_level > risk_moderate_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY actions

    # Priority 2: Trend Following (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level <= risk_moderate_threshold:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20)  # Reward for BUY signals in an uptrend
        elif trend_direction < 0:  # Downtrend
            reward += np.random.uniform(10, 20)  # Reward for SELL signals in a downtrend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < risk_moderate_threshold:
        if enhanced_s[123] < lower_band:  # Oversold condition
            reward += 15.0  # Reward for buying an oversold signal
        elif enhanced_s[123] > upper_band:  # Overbought condition
            reward += 15.0  # Reward for selling an overbought signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level <= risk_moderate_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50% during high volatility

    return float(np.clip(reward, -100, 100))  # Clamp the reward between -100 and 100