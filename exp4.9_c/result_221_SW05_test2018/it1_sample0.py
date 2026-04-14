import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price momentum (current price vs. moving average of last 5 days)
    moving_average = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average) / (moving_average if moving_average != 0 else 1)

    # Feature 2: Price volatility (standard deviation of price over last 20 days)
    price_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0

    # Feature 3: Average volume over the last 20 days
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]

    # Feature 4: Volume change (current volume vs. average volume)
    volume_change = (volumes[-1] - avg_volume) / (avg_volume if avg_volume != 0 else 1)

    return np.array([price_momentum, price_volatility, volume_change])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    price_momentum = features[0]
    price_volatility = features[1]

    reward = 0.0
    
    # Calculate dynamic thresholds based on historical data (assuming a historical context available)
    risk_threshold_high = 0.7  # Example threshold for high risk, can be adjusted based on historical std
    trend_threshold = 0.3       # Example threshold for trend recognition
    volatility_threshold = 0.6   # Example threshold for high volatility, can be adjusted
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        # Strong negative reward for BUY-aligned features
        if price_momentum > 0:
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        elif price_momentum < 0:
            reward += np.random.uniform(5, 10)

    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if price_momentum > 0:
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > trend_threshold and price_momentum > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -trend_threshold and price_momentum < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if price_momentum < 0:  # Overbought and bearish signal
            reward += np.random.uniform(10, 20)
        elif price_momentum > 0:  # Oversold and bullish signal
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within [-100, 100]