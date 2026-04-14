import numpy as np

def revise_state(s):
    # Extract closing prices
    closing_prices = s[0::6]  # Closing prices are at indices 0, 6, 12, ..., 114
    
    # Number of observations
    N = len(closing_prices)
    
    # Feature 1: 10-Day Moving Average
    ma_10 = np.mean(closing_prices[-10:]) if N >= 10 else np.nan
    
    # Feature 2: 20-Day Moving Average
    ma_20 = np.mean(closing_prices[-20:]) if N >= 20 else np.nan
    
    # Feature 3: 14-Day Relative Strength Index (RSI)
    if N >= 14:
        delta = np.diff(closing_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs = avg_gain / (avg_loss + 1e-8)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.nan
    
    # Feature 4: Price Momentum (Current Price - Previous Day's Price)
    price_momentum = closing_prices[-1] - closing_prices[-2] if N >= 2 else np.nan
    
    # Feature 5: Standard Deviation of Returns (Volatility)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-20:]) if N >= 20 else np.nan
    
    # Feature 6: Volume Change Percentage
    volumes = s[4::6]  # Extract trading volumes
    volume_changes = np.diff(volumes) / volumes[:-1] if len(volumes) > 1 else np.nan
    volume_change_percentage = volume_changes[-1] if len(volume_changes) > 0 else np.nan
    
    # Combine features into a single array
    features = [ma_10, ma_20, rsi, price_momentum, volatility, volume_change_percentage]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # Extract regime vector
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for BUY-aligned features
        return np.clip(reward, -100, 100)  # Early return if risk is high
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 30  # Positive reward for bullish trend
        else:
            reward += 30  # Positive reward for bearish trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 0:  # Assuming oversold condition
            reward += 20  # Reward for buying during mean-reversion
        else:
            reward -= 20  # Penalize selling during mean-reversion

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > np.std(enhanced_s[123:]) and risk_level < 0.4:  # Use relative threshold
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)