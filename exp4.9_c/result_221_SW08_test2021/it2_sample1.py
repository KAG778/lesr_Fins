import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

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
    high_prices = s[2::6]
    low_prices = s[3::6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr) if len(tr) > 0 else 0

    features = [price_change_pct, macd, historical_vol, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Assuming features start from index 123
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals in high risk
        reward += 10   # Mild positive for SELL signals
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 20  # Reward for bullish momentum
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 20  # Reward for bearish momentum

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 10  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the specified range
    reward = np.clip(reward, -100, 100)

    return reward