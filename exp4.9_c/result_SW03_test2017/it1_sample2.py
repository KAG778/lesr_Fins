import numpy as np

def revise_state(s):
    features = []
    
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]         # Extract trading volumes

    # Feature 1: Price Momentum (Rate of Change)
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(momentum)

    # Feature 2: Average Trading Volume Change (last 5 days)
    avg_volume_today = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    avg_volume_past = np.mean(volumes[-10:-5]) if len(volumes) >= 10 else 0
    volume_change = (avg_volume_today - avg_volume_past) / avg_volume_past if avg_volume_past != 0 else 0
    features.append(volume_change)

    # Feature 3: Price Action (Close - Open)
    close_today = closing_prices[-1]
    open_today = s[1::6][-1]  # Extract the opening price of the most recent day
    price_action = (close_today - open_today) / open_today if open_today != 0 else 0
    features.append(price_action)

    # Feature 4: Historical Volatility (20-day)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns) * np.sqrt(20)  # Annualized volatility
    features.append(historical_volatility)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate thresholds based on historical data
    # For simplicity, we will assume we have some historical data to compute these thresholds
    historical_volatility_threshold = 0.2  # Example threshold based on historical data
    risk_threshold_upper = 0.7
    risk_threshold_lower = 0.4
    trend_threshold = 0.3
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_upper:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features
    elif risk_level > risk_threshold_lower:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_lower:
        momentum = enhanced_s[123]  # Assuming features[0] is momentum
        if trend_direction > 0:
            reward += 20 * momentum  # Positive reward for upward momentum
        else:
            reward += 10 * (1 - momentum)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > historical_volatility_threshold and risk_level < risk_threshold_lower:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within [-100, 100]
    return np.clip(reward, -100, 100)