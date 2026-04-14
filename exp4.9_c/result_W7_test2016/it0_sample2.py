import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract volumes
    adjusted_closing_prices = s[5::6]  # Extract adjusted closing prices

    # Calculate daily returns
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.append(0, daily_returns)  # Padding for alignment

    # Feature 1: Average Daily Return over the last 20 days
    avg_daily_return = np.mean(daily_returns)

    # Feature 2: Volatility (standard deviation of returns)
    volatility = np.std(daily_returns)

    # Feature 3: Current Price Momentum (current closing price - previous closing price)
    momentum = closing_prices[-1] - closing_prices[-2]  # Last day momentum

    # Feature 4: Price Range (high - low)
    price_range = np.max(high_prices) - np.min(low_prices)

    # Compile features into a single array
    features = [avg_daily_return, volatility, momentum, price_range]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Random strong negative reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Trend is up, reward for positive momentum
            reward += np.random.uniform(10, 20)  # Reward for upward features
        elif trend_direction < -0.3:
            # Trend is down, reward for negative momentum
            reward += np.random.uniform(10, 20)  # Reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features
        reward += np.random.uniform(5, 15)  # Reward for correct mean-reversion

    # Priority 4 — HIGH VOLATILITY (no crisis)
    if volatility_level > 0.6 and risk_level < 0.4:
        # Reduce reward magnitude by 50% for uncertainty 
        reward *= 0.5

    # Ensure the reward is capped within [-100, 100]
    return np.clip(reward, -100, 100)