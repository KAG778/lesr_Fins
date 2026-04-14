import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]
    volumes = s[4:120:6]

    # Feature 1: Price Change Percentage from 10 days ago
    if len(closing_prices) > 10 and closing_prices[10] != 0:
        price_change_percentage = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11]
    else:
        price_change_percentage = 0
    features.append(price_change_percentage)

    # Feature 2: Historical Volatility (30-day)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns) * np.sqrt(30) if len(returns) > 0 else 0
    features.append(historical_volatility)

    # Feature 3: Momentum (Rate of Change over 5 days)
    if len(closing_prices) > 5 and closing_prices[-6] != 0:
        momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        momentum = 0
    features.append(momentum)

    # Feature 4: Average True Range (ATR) over the last 14 days
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    if len(high_prices) >= 14 and len(low_prices) >= 14:
        tr = np.maximum(high_prices[-14:] - low_prices[-14:], 
                        np.maximum(np.abs(high_prices[-14:] - closing_prices[-15:-1]), 
                                   np.abs(low_prices[-14:] - closing_prices[-15:-1])))
        atr = np.mean(tr)
    else:
        atr = 0
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Calculate dynamic thresholds based on historical standard deviation of risk levels
    historical_risk_levels = enhanced_s[123:]  # Assuming these are historical risk levels
    risk_std = np.std(historical_risk_levels)
    high_risk_threshold = 0.7 * risk_std
    low_risk_threshold = 0.4 * risk_std
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY-aligned features
        reward += np.random.uniform(5, 15)    # Mild positive for SELL-aligned features
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20 * abs(trend_direction)  # Reward momentum alignment based on strength of trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.mean(historical_risk_levels) * 1.5:  # Relative measure
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within specified range