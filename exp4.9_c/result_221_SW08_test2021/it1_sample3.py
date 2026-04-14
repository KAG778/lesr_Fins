import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    # Feature 1: Price Momentum (percentage change over the last 3 days)
    price_momentum = (closing_prices[-1] - closing_prices[-4]) / closing_prices[-4] if closing_prices[-4] != 0 else 0

    # Feature 2: Average Volume Change (percentage change from the last period)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    # Feature 3: Volatility (standard deviation of returns over the last 20 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Calculate daily returns
    volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0

    # Feature 4: Enhanced Relative Strength Index (RSI) with dynamic period based on volatility
    def compute_dynamic_rsi(prices, period):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        average_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        average_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = average_gain / average_loss if average_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_dynamic_rsi(closing_prices, max(14, int(volatility * 100)))  # Dynamic period based on volatility

    features = [price_momentum, volume_change, volatility, rsi]
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
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the specified range
    reward = np.clip(reward, -100, 100)

    return float(reward)