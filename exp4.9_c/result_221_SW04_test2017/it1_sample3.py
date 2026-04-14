import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    trading_volumes = s[4:120:6]  # Extracting trading volumes
    
    # Feature 1: Price Change Ratio (current price - previous price) / previous price
    price_change_ratio = np.zeros(len(closing_prices))
    for i in range(1, len(closing_prices)):
        if closing_prices[i-1] != 0:  # Prevent division by zero
            price_change_ratio[i] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
    
    # Feature 2: Average Trading Volume over the last 20 days
    average_volume = np.mean(trading_volumes[-20:]) if len(trading_volumes) >= 20 else 0

    # Feature 3: Exponential Moving Average (EMA) of closing prices over 10 days
    if len(closing_prices) >= 10:
        ema = np.mean(closing_prices[-10:])  # Simple EMA for demonstration
    else:
        ema = np.nan  # Handle edge case

    # Feature 4: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0

    # Compile features
    features = [price_change_ratio[-1], average_volume, ema, atr]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_volatility = np.std(features[3])  # Assuming features[3] is ATR
    mean_volume = np.mean(features[1]) if features[1] > 0 else 1e-5  # Avoid division by zero

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 40 * (features[0] > 0)  # Strong negative for BUY signals
        reward += 10 * (features[0] < 0)   # Mild positive for SELL signals
    elif risk_level > 0.4:
        reward -= 20 * (features[0] > 0)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if np.abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 30 * (features[0] > 0)  # Positive reward for upward momentum
        reward += 10 * (features[0] < 0)  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if np.abs(trend_direction) < 0.3:
        reward += 15 * (features[0] < 0)  # Reward for mean-reversion BUY signals
        reward += 15 * (features[0] > 0)  # Reward for mean-reversion SELL signals

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds