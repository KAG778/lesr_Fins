import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    trading_volumes = s[4:120:6]  # Extracting trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) for 20 days
    if len(closing_prices) >= 20:
        weights = np.exp(np.linspace(-1, 0, 20))
        weights /= weights.sum()
        ema_20 = np.dot(weights, closing_prices[-20:])
    else:
        ema_20 = np.nan  # Handle edge case, not enough data

    # Feature 2: Price Momentum (current price - EMA)
    price_momentum = closing_prices[-1] - ema_20 if not np.isnan(ema_20) else np.nan

    # Feature 3: Historical Volatility (using log returns)
    returns = np.diff(np.log(closing_prices))
    historical_volatility = np.std(returns[-30:]) if len(returns) >= 30 else np.nan  # 30-day historical volatility

    # Feature 4: Average Volume Change (compared to previous 20 days)
    if len(trading_volumes) >= 20:
        avg_volume_past = np.mean(trading_volumes[-20:])
        avg_volume_current = np.mean(trading_volumes[-20-20:-20]) if len(trading_volumes) > 40 else 0
        volume_change = (avg_volume_current - avg_volume_past) / avg_volume_past if avg_volume_past != 0 else 0
    else:
        volume_change = np.nan

    features = [price_momentum, historical_volatility, volume_change]
    
    # Filter out NaN values
    return np.nan_to_num(np.array(features))

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Calculate relative thresholds for risk management
    historical_volatility = np.std(enhanced_s[123:])  # Using calculated features as proxy for historical volatility
    high_risk_threshold = 1.5 * historical_volatility  # Example threshold for high risk
    low_risk_threshold = 0.5 * historical_volatility   # Example threshold for low risk

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for buying in high-risk
    elif risk_level > low_risk_threshold:
        reward -= 20  # Mild negative for buying in moderate-risk
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        reward += 20 * (1 if trend_direction > 0 else -1)  # Positive for aligning with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 10  # Reward for mean-reversion strategies

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds