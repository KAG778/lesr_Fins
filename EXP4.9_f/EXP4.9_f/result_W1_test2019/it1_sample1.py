import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []
    
    # Extract closing prices and trading volumes
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    volumes = s[4::6]         # Every 6th element starting from index 4
    
    # Feature 1: Price Momentum (current closing - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(price_momentum)

    # Feature 2: Average Daily Return over the last 20 days
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
    features.append(avg_daily_return)

    # Feature 3: Volatility (Standard Deviation of daily returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
    features.append(volatility)

    # Feature 4: Average Volume over the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0
    features.append(avg_volume)

    # Feature 5: Strength of the trend (using a simple moving average)
    sma_20 = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    trend_strength = (closing_prices[-1] - sma_20) / sma_20 if sma_20 != 0 else 0
    features.append(trend_strength)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate relative thresholds based on historical data (e.g., mean and std)
    historical_risk_threshold = 0.5  # Example relative threshold, needs to be adjusted based on historical data
    historical_volatility_threshold = 0.5  # Likewise for volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk_threshold:
        reward += -50  # Strong negative for BUY-aligned features
        reward += 10   # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < historical_risk_threshold:
        if trend_direction > 0:
            reward += 20  # Strong positive for upward trends
        else:
            reward += 20  # Strong positive for downward trends

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion strategies

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility_threshold and risk_level < historical_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds