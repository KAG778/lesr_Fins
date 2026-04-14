import numpy as np

def revise_state(s):
    # Extracting closing prices and trading volumes
    closing_prices = s[0:120:6]  # Closing prices
    trading_volumes = s[4:120:6]  # Trading volumes

    # Feature 1: Price Change Ratio (current price - previous price) / previous price
    price_change_ratio = np.zeros(len(closing_prices) - 1)
    for i in range(1, len(closing_prices)):
        if closing_prices[i - 1] != 0:  # Prevent division by zero
            price_change_ratio[i - 1] = (closing_prices[i] - closing_prices[i - 1]) / closing_prices[i - 1]
    
    # Feature 2: Average Trading Volume over the last 20 days
    average_volume = np.mean(trading_volumes)

    # Feature 3: 14-day Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0).mean()  # Average gain
    loss = -np.where(deltas < 0, deltas, 0).mean()  # Average loss
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 4: 20-day Historical Volatility
    returns = np.diff(np.log(closing_prices))
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

    # Feature 5: Price to Moving Average Ratio (20-day MA)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    price_to_ma_ratio = closing_prices[-1] / moving_average if moving_average != 0 else 0

    features = [
        price_change_ratio[-1],  # Last day's price change ratio
        average_volume,           # Average volume
        rsi,                      # RSI
        historical_volatility,    # Historical volatility
        price_to_ma_ratio        # Price to MA ratio
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]

    # Dynamic thresholds based on historical volatility
    risk_threshold = 0.5  # Example threshold for risk level
    high_vol_threshold = 0.6 * np.std(features[3])  # Volatility based on historical std
    
    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        # Strong negative reward for buying in high risk
        reward -= np.clip(50 * features[0], 0, 100)  # Severity based on price change ratio
        reward += np.clip(15 * features[1], 0, 30)   # Mild positive for selling in high risk

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 30 * features[0]  # Strong positive for bullish trends
        elif trend_direction < 0:
            reward += 30 * features[1]  # Strong positive for bearish trends

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # RSI indicating oversold
            reward += 20  # Reward for buying in oversold conditions
        if features[2] > 70:  # RSI indicating overbought
            reward += 20  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > high_vol_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds