import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]          # Extracting trading volumes

    # Feature 1: Price Change Percentage over the last 5 days
    price_change_pct = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Feature 2: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:])  # Short-term average
    long_ema = np.mean(closing_prices[-26:])   # Long-term average
    macd = short_ema - long_ema

    # Feature 3: Historical Volatility (standard deviation of returns over the last 20 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    historical_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0

    # Feature 4: Average True Range (ATR) as a measure of market volatility
    high_low = np.array([s[i*6+2] - s[i*6+3] for i in range(20)])  # High - Low
    high_close = np.array([abs(s[i*6+2] - s[i*6+1]) for i in range(20)])  # High - Previous Close
    low_close = np.array([abs(s[i*6+3] - s[i*6+1]) for i in range(20)])  # Low - Previous Close
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = np.mean(tr) if len(tr) > 0 else 0

    features = [price_change_pct, macd, historical_vol, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Dynamic thresholds (based on historical std deviation)
    risk_threshold_high = 0.7
    risk_threshold_medium = 0.4
    
    # Calculate rewards
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY signals
        reward += 10 * np.random.uniform(0, 1)  # Mild positive for SELL signals
    elif risk_level > risk_threshold_medium:
        reward -= 30  # Moderate negative reward for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Positive reward based on trend direction

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is bounded within [-100, 100]
    reward = np.clip(reward, -100, 100)

    return reward