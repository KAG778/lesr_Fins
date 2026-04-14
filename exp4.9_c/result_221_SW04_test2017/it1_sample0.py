import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    trading_volumes = s[4:120:6]  # Extract trading volumes
    days = len(closing_prices)

    # Feature 1: 14-day RSI
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 2: 5-day Moving Average
    moving_average_5 = np.mean(closing_prices[-5:]) if days >= 5 else np.nan

    # Feature 3: 30-day Historical Volatility
    returns = np.diff(closing_prices)
    historical_volatility = np.std(returns[-30:]) if len(returns) >= 30 else np.nan

    # Feature 4: Price Momentum (current - 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if days > 6 else 0

    # Feature 5: Volume Change Ratio (current volume - average volume)
    avg_volume = np.mean(trading_volumes) if len(trading_volumes) > 0 else 1
    volume_change_ratio = trading_volumes[-1] / avg_volume if avg_volume > 0 else 0

    features = [rsi, moving_average_5, historical_volatility, price_momentum, volume_change_ratio]
    
    return np.nan_to_num(np.array(features))  # Ensure no NaN values

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0
    
    # Calculate relative thresholds based on historical std
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    volatility_threshold_high = 0.6
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= np.clip(50 * features[0], 30, 50)  # Strong negative for high-risk buying
        reward += np.clip(5 * features[-1], 5, 10)   # Mild positive for selling (high volume)
    elif risk_level > risk_threshold_medium:
        reward -= 20 * features[0]  # Moderate negative for buying

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:
            reward += 20 * features[3]  # Positive reward for positive momentum
        elif trend_direction < 0:
            reward += 20 * -features[3]  # Positive reward for negative momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # Oversold condition
            reward += 15  # Reward for buying in oversold conditions
        if features[0] > 70:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold_high and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds