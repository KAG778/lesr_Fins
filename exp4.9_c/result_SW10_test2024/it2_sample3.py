import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    features = []
    
    # 1. Crisis Indicator (percentage drop from peak in the last 20 days)
    peak_price = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    current_price = closing_prices[-1]
    crisis_indicator = ((peak_price - current_price) / peak_price) * 100  # Percentage drop
    features.append(crisis_indicator)

    # 2. 20-day historical volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
    features.append(historical_volatility)

    # 3. Momentum based on Rate of Change (ROC) for the last 5 days
    if len(closing_prices) >= 6:
        roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    else:
        roc = 0
    features.append(roc)

    # 4. Average trading volume over the last 20 days
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    features.append(avg_volume)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds for dynamic reward mechanism
    historical_std = np.std(enhanced_s[123:])  # Use feature array for volatility
    high_vol_threshold = historical_std * 1.5  # Example condition for high volatility
    low_vol_threshold = historical_std * 0.5  # Example condition for low volatility

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10 if trend_direction < 0 else -10  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 20 * np.sign(trend_direction)  # Strong reward for trend alignment

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for potential mean-reversion features
        reward -= 10  # Penalize for chasing breakouts

    # Priority 4: HIGH VOLATILITY
    if volatility_level > high_vol_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%
    elif volatility_level < low_vol_threshold:
        reward *= 1.5  # Potentially increase reward if volatility is low

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds