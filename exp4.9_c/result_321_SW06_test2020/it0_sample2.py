import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Calculate daily returns
    closing_prices = s[0::6]  # Get closing prices
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    
    # Feature 1: Average Daily Return over the last 19 days
    avg_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
    features.append(avg_daily_return)

    # Feature 2: Volatility (Standard deviation of daily returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 0 else 0
    features.append(volatility)

    # Feature 3: Relative Strength Index (RSI) over the last 14 days
    # Calculate gains and losses
    gains = (daily_returns[daily_returns > 0]).sum() / 14 if len(daily_returns[daily_returns > 0]) > 0 else 0
    losses = (-daily_returns[daily_returns < 0]).sum() / 14 if len(daily_returns[daily_returns < 0]) > 0 else 0
    
    # Calculate RS and RSI
    rs = gains / losses if losses > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

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

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward += -40  # Arbitrary strong negative value
        # MILD POSITIVE reward for SELL-aligned features
        reward += 5    # Mild positive value for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Arbitrary moderate negative value

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 15  # Positive reward for bullish trend
        elif trend_direction < -0.3:
            reward += 15  # Positive reward for bearish trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clipping the reward to the range [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward