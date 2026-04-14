import numpy as np

def revise_state(s):
    # Extract relevant components from the raw state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes
    high_prices = s[2::6]      # High prices
    low_prices = s[3::6]       # Low prices

    # Feature 1: 10-Day Moving Average of Closing Prices
    ma_window = 10
    if len(closing_prices) >= ma_window:
        moving_average = np.mean(closing_prices[-ma_window:])
    else:
        moving_average = closing_prices[-1]  # Fallback to last closing price

    # Feature 2: Price Momentum (current close - close 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else 0.0

    # Feature 3: Volatility (standard deviation of returns over the last 10 days)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-ma_window:]) if len(returns) >= ma_window else 0.0

    # Feature 4: Volume Change (current volume - average volume of last 10 days)
    avg_volume = np.mean(volumes[-ma_window:]) if len(volumes) >= ma_window else 1.0  # Avoid division by zero
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume > 0 else 0.0

    # Feature 5: Price Range (high - low over the last 10 days)
    price_range = np.max(high_prices[-ma_window:]) - np.min(low_prices[-ma_window:]) if len(high_prices) >= ma_window else 0.0

    features = [moving_average, price_momentum, volatility, volume_change, price_range]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY signals
        reward += 10.0 * features[3]  # Mild positive for SELL-aligned features (volume change)
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[1] > 0:  # Positive momentum
            reward += trend_direction * features[1] * 10.0  # Strong reward for aligning with trend
        elif features[1] < 0:  # Negative momentum
            reward += trend_direction * features[1] * 10.0  # Reward for correct bearish bet

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 0:  # Oversold condition
            reward += 5.0  # Positive reward for potential buy
        elif features[1] > 0:  # Overbought condition
            reward -= 5.0  # Negative reward for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))