import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Closing prices from raw state
    opening_prices = s[1::6]  # Opening prices from raw state
    high_prices = s[2::6]     # High prices from raw state
    low_prices = s[3::6]      # Low prices from raw state
    volumes = s[4::6]         # Trading volumes from raw state

    # Feature 1: Daily Return (percentage change from previous closing price)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    # Padding with NaN for the first day where we can't compute a return
    daily_returns = np.insert(daily_returns, 0, np.nan)

    # Feature 2: Moving Average (10-day simple moving average of closing prices)
    moving_average = np.convolve(closing_prices, np.ones(10)/10, mode='valid')
    # Padding to keep the length consistent with original state
    moving_average = np.concatenate((np.full(9, np.nan), moving_average))

    # Feature 3: Price Range (High - Low for each day)
    price_range = high_prices - low_prices

    # Handle edge cases with NaN values 
    # Replace NaN with 0 or appropriate value based on context
    daily_returns = np.nan_to_num(daily_returns, nan=0.0)
    moving_average = np.nan_to_num(moving_average, nan=0.0)

    # Combine features into a single array
    features = [daily_returns[-1], moving_average[-1], price_range[-1]]

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward += -np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward += -np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward += -np.random.uniform(5, 15)  # Penalize breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]