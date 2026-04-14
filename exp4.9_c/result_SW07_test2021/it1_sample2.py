import numpy as np

def revise_state(s):
    # Extract closing prices, volumes, and calculate features
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes
    high_prices = s[2::6]      # High prices
    low_prices = s[3::6]       # Low prices

    # Feature 1: Relative Strength Index (RSI) - captures overbought/oversold conditions
    deltas = np.diff(closing_prices)
    gain = (deltas[deltas > 0]).sum() / 14  # Average gain over 14 days
    loss = (-deltas[deltas < 0]).sum() / 14  # Average loss over 14 days
    rs = gain / loss if loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    # Feature 2: Bollinger Bands Width - measures volatility
    moving_avg = np.mean(closing_prices[-20:])  # 20-day moving average
    std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
    bollinger_band_width = std_dev / moving_avg if moving_avg != 0 else 0

    # Feature 3: Momentum - rate of change in prices over the last 5 days
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0

    # Collect features into a list
    features = [rsi, bollinger_band_width, momentum]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate relative thresholds based on historical data
    risk_threshold = 0.5  # Placeholder, should be based on historical std
    trend_threshold = 0.3  # Placeholder, should be based on historical std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY-aligned features
        reward += np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL-aligned features
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING (when risk is low)
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]