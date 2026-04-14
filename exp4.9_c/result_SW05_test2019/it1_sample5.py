import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    recent_momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0
    price_momentum_history = closing_prices[-20:]  # Last 20 closing prices for historical context
    historical_mean = np.mean(price_momentum_history)
    historical_std = np.std(price_momentum_history)
    
    # Z-Score for Price Momentum
    z_momentum = (recent_momentum - historical_mean) / historical_std if historical_std > 0 else 0
    
    # Calculate volatility over the last 5 closing prices
    volatility = np.std(closing_prices[-5:]) if len(closing_prices) > 5 else 0
    volatility_history = closing_prices[-20:]  # Historical context for volatility
    historical_vol_mean = np.mean(volatility_history)
    historical_vol_std = np.std(volatility_history)
    
    # Z-Score for Volatility
    z_volatility = (volatility - historical_vol_mean) / historical_vol_std if historical_vol_std > 0 else 0
    
    # ADX Calculation (using a simple method for demonstration)
    # Here, we will use a simplified approach to calculate ADX based on price changes
    if len(closing_prices) > 14:
        highs = s[2:120:6]
        lows = s[3:120:6]
        plus_dm = np.mean(np.maximum(highs[1:] - highs[:-1], 0))
        minus_dm = np.mean(np.maximum(lows[:-1] - lows[1:], 0))
        adx = (plus_dm - minus_dm) / (plus_dm + minus_dm) if (plus_dm + minus_dm) != 0 else 0
    else:
        adx = 0

    features = [z_momentum, z_volatility, adx]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Dynamic thresholds based on historical data (could be enhanced with actual historical data)
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    momentum_threshold = 0.3
    volatility_threshold = 0.6

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY
    elif risk_level > risk_threshold_medium:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > momentum_threshold and risk_level < risk_threshold_medium:
        if trend_direction > 0:
            reward += 20  # Strong positive reward for buying in an uptrend
        elif trend_direction < 0:
            reward += 20  # Strong positive reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) <= momentum_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds