import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    trading_volumes = s[4:120:6]  # Extract trading volumes

    # Feature 1: 14-day Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 2: 20-day Moving Average
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0

    # Feature 3: Price Momentum (current price - previous price)
    price_momentum = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0
    
    # Feature 4: Average True Range (ATR) for volatility measurement
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0

    # Feature 5: Recent Price Change Ratio
    price_change_ratio = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    features = [rsi, moving_average, price_momentum, atr, price_change_ratio]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(features)  # Using features to assess standard deviation
    risk_threshold_high = 0.5 + historical_std  # Example threshold for high risk
    risk_threshold_medium = 0.5  # Example threshold for moderate risk
    volatility_threshold = 0.6  # Example threshold for high volatility

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50 * (features[4] > 0)  # Strong negative for BUY signals in high risk
        reward += 10 * (features[4] < 0)   # Mild positive for SELL signals in high risk
    elif risk_level > risk_threshold_medium:
        reward -= 20 * (features[4] > 0)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        reward += 20 * features[2]  # Reward for price momentum aligning with trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # Oversold condition
            reward += 15  # Reward for buying in oversold conditions
        if features[0] > 70:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds