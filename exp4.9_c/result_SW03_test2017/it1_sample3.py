import numpy as np

def revise_state(s):
    features = []
    
    # Feature 1: Price Momentum (Rate of Change over the last 5 days)
    momentum = (s[114] - s[84]) / s[84] if s[84] != 0 else 0  # Current close vs close 5 days ago
    features.append(momentum)

    # Feature 2: Average Trading Volume over the last 5 days
    avg_volume = np.mean(s[4::6][-5:])  # Last 5 volumes
    features.append(avg_volume)

    # Feature 3: Historical Volatility (20-day)
    returns = np.diff(s[0::6]) / s[0::6][:-1]  # Daily returns
    historical_volatility = np.std(returns) * np.sqrt(20)  # Annualized volatility
    features.append(historical_volatility)
    
    # Feature 4: Price Change Percentage from 5 days ago
    price_change_percentage = (s[114] - s[84]) / s[84] if s[84] != 0 else 0  # Current close vs close 5 days ago
    features.append(price_change_percentage)

    # Feature 5: Relative Strength Index (RSI) over the last 14 days
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Positive trend
            reward += 10 * (enhanced_s[123][0] + 1)  # Assuming features[0] indicates momentum
        else:  # Negative trend
            reward += 10 * (1 - enhanced_s[123][0])  # Assuming features[0] indicates momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += 10 * (1 - enhanced_s[123][0])  # Assuming features[0] indicates mean-reversion potential

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]