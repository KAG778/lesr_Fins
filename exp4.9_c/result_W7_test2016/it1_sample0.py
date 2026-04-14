import numpy as np

def revise_state(s):
    # s: 120d raw state
    n_days = 20
    closing_prices = s[0::6][:n_days]  # Extract closing prices
    volumes = s[4::6][:n_days]  # Extract trading volumes

    # Feature 1: Average Daily Return over the last 'n_days'
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0

    # Feature 2: Volatility (Standard Deviation of daily returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0

    # Feature 3: Current Price Momentum (current closing price - moving average of last 5 days)
    moving_average_5 = np.mean(closing_prices[-5:]) if n_days >= 5 else closing_prices[-1]
    momentum = closing_prices[-1] - moving_average_5

    # Feature 4: Volume Spike (current volume / average volume of last 5 days)
    avg_volume_5 = np.mean(volumes[-5:]) if n_days >= 5 else volumes[-1]
    volume_spike = volumes[-1] / avg_volume_5 if avg_volume_5 > 0 else 0

    features = [avg_daily_return, volatility, momentum, volume_spike]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate thresholds based on historical data (assuming we have historical std dev from past results)
    historical_volatility = 0.2  # Placeholder for historical volatility threshold (to be calculated externally)
    historical_risk = 0.5  # Placeholder for historical risk threshold (to be calculated externally)

    # Priority 1 — RISK MANAGEMENT
    if risk_level > historical_risk:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # MILD POSITIVE reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < historical_risk:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Positive reward for upward momentum
        elif trend_direction < 0:
            reward += np.random.uniform(10, 20)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features
        reward -= np.random.uniform(5, 10)   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility and risk_level < historical_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds