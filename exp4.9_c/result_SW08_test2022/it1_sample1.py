import numpy as np

def revise_state(s):
    features = []

    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]          # Trading volumes

    # Feature 1: Average Daily Return (for the last 20 days)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    daily_returns = np.concatenate(([0], daily_returns))  # Prepend 0 for alignment
    avg_daily_return = np.mean(daily_returns[-20:]) if len(daily_returns) > 20 else 0
    features.append(avg_daily_return)

    # Feature 2: Volatility (standard deviation of daily returns over last 20 days)
    volatility = np.std(daily_returns[-20:]) if len(daily_returns) > 20 else 0
    features.append(volatility)

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(daily_returns > 0, daily_returns, 0)
    losses = np.where(daily_returns < 0, -daily_returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss > 0 else 100
    features.append(rsi)

    # Feature 4: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0
    features.append(price_momentum)

    # Feature 5: Average Trading Volume (last 20 days)
    avg_volume = np.mean(volumes[-20:]) if len(volumes) > 20 else 0
    features.append(avg_volume)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Define historical thresholds for risk levels
    risk_threshold_high = 0.7
    risk_threshold_moderate = 0.4

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative reward for BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features
    elif risk_level > risk_threshold_moderate:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 15  # Positive reward for upward features
        elif trend_direction < 0:  # Downtrend
            reward += 15  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within bounds
    return np.clip(reward, -100, 100)