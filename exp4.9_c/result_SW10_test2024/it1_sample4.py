import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    features = []

    # Feature 1: 20-day historical volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
    features.append(historical_volatility)

    # Feature 2: Crisis Indicator (percentage drop from peak in the last 20 days)
    peak_price = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    current_price = closing_prices[-1]
    crisis_indicator = ((peak_price - current_price) / peak_price) * 100  # Percentage drop
    features.append(crisis_indicator)

    # Feature 3: 20-day average volume
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    features.append(avg_volume)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE for BUY-aligned features
        reward += 10 if trend_direction < 0 else -10  # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 25 if trend_direction > 0 else 15  # Strong reward for upward momentum, moderate for downward

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 20  # Reward potential mean-reversion features
        reward -= 10  # Penalize for chasing breakouts

    # Priority 4: HIGH VOLATILITY
    if volatility_level > np.std(enhanced_s[123:]) * 0.5 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds