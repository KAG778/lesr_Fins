import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extracting closing prices
    trading_volumes = s[4:120:6]  # Extracting trading volumes
    days = len(closing_prices)

    # Feature 1: Weighted Moving Average (WMA) - 20 days
    if days >= 20:
        weights = np.arange(1, 21)
        wma = np.dot(weights, closing_prices[-20:]) / weights.sum()
    else:
        wma = np.nan
    
    # Feature 2: Average True Range (ATR)
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0

    # Feature 3: Price to WMA Ratio
    price_to_wma_ratio = closing_prices[-1] / wma if wma != 0 else 0

    # Feature 4: 14-day Relative Strength Index (RSI)
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = -np.where(deltas < 0, deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 5: Price Change Ratio (current price - previous price) / previous price
    price_change_ratio = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if days > 1 else 0

    features = [wma, atr, price_to_wma_ratio, rsi, price_change_ratio]
    
    return np.nan_to_num(np.array(features))  # Replace NaNs with 0 to ensure usability

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    
    # Calculate dynamic thresholds based on historical data (using features)
    historical_std = np.std(features)  # Standard deviation of features
    risk_threshold_high = 0.7 * historical_std  # Dynamic high-risk threshold
    risk_threshold_medium = 0.4 * historical_std  # Dynamic medium-risk threshold

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals in high risk
        reward += 10 * (features[4] < 0)  # Reward for SELL signals if price change ratio is negative
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if np.abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0:  # Uptrend
            reward += 20 * features[4]  # Positive reward for price change ratio in uptrend
        else:  # Downtrend
            reward += -20 * features[4]  # Negative reward for price change ratio in downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < 30:  # Oversold condition
            reward += 15  # Reward for buying in oversold conditions
        if features[3] > 70:  # Overbought condition
            reward += 15  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds