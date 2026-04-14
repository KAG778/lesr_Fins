import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    volumes = s[4::6]         # Extracting trading volumes

    # Feature 1: Rate of Change (ROC) over the last 10 days
    roc = (closing_prices[-1] - closing_prices[-11]) / closing_prices[-11] if len(closing_prices) >= 11 and closing_prices[-11] != 0 else 0

    # Feature 2: Historical Volatility as the standard deviation of returns
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

    # Feature 3: Average Volume over the last 10 days
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else 0

    # Feature 4: Bollinger Bands Width (using 20-day SMA)
    if len(closing_prices) >= 20:
        sma = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        bollinger_width = (std_dev / sma) if sma != 0 else 0
    else:
        bollinger_width = 0

    features = [roc, historical_volatility, avg_volume, bollinger_width]
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
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Positive reward based on trend direction

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward stays within [-100, 100]
    return np.clip(reward, -100, 100)