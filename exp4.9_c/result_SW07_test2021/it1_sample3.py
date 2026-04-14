import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Closing prices for the last 20 days
    volumes = s[4::6]          # Trading volumes for the last 20 days

    # Feature 1: Relative Momentum (current momentum vs historical average)
    price_momentum = closing_prices[-1] - closing_prices[-6]
    historical_momentum = np.mean(closing_prices[-6:] - closing_prices[-12:-6])  # Compare to last 6 days
    relative_momentum = price_momentum / (historical_momentum if historical_momentum != 0 else 1)

    # Feature 2: Relative Volatility (current volatility vs historical average)
    historical_volatility = np.std(closing_prices[-20:])  # Volatility over past 20 days
    current_volatility = np.std(np.diff(closing_prices[-5:]))  # Volatility over the last 5 days
    relative_volatility = current_volatility / (historical_volatility if historical_volatility != 0 else 1)

    # Feature 3: Distance from Historical Moving Average (mean reversion)
    moving_average = np.mean(closing_prices[-20:])
    distance_from_ma = closing_prices[-1] - moving_average  # Current price - MA

    # Combine features into a single array
    return np.array([relative_momentum, relative_volatility, distance_from_ma])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_risk_threshold = 0.6  # Example threshold; this should be calculated from historical data
    historical_trend_threshold = 0.3  
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk_threshold:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY signals
        reward += np.random.uniform(5, 10)    # MILD POSITIVE for SELL signals

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > historical_trend_threshold and risk_level < 0.4:
        if trend_direction > historical_trend_threshold:  # Strong uptrend
            reward += np.random.uniform(15, 25)  # Positive reward for upward momentum
        elif trend_direction < -historical_trend_threshold:  # Strong downtrend
            reward += np.random.uniform(15, 25)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < historical_trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean reversion features
        reward -= np.random.uniform(5, 15)  # Penalize if chasing a breakout

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds