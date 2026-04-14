import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Closing prices (every 6th element starting from index 0)
    volumes = s[4:120:6]          # Trading volumes (every 6th element starting from index 4)

    # Feature 1: Price Change Percentage
    price_change_pct = np.zeros(19)  # We have 20 days, so 19 changes
    for i in range(1, 20):
        if closing_prices[i-1] != 0:  # Avoid division by zero
            price_change_pct[i-1] = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
        else:
            price_change_pct[i-1] = 0  # No change if previous price is zero

    # Feature 2: Average Volume
    average_volume = np.mean(volumes)

    # Feature 3: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_value = calculate_rsi(closing_prices)

    # Return computed features
    return np.array([price_change_pct[-1], average_volume, rsi_value])  # Return only the last price change percentage

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Extract features
    features = enhanced_s[123:]
    last_price_change_pct = features[0]
    average_volume = features[1]
    rsi_value = features[2]

    # Initialize reward
    reward = 0.0

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        if last_price_change_pct > 0:  # Buy-aligned features
            reward = np.random.uniform(-50, -30)
        else:  # Sell-aligned features
            reward = np.random.uniform(5, 10)
    elif risk_level > 0.4:
        if last_price_change_pct > 0:  # Buy signals
            reward = np.random.uniform(-10, -5)

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and last_price_change_pct > 0:  # Uptrend and positive change
            reward += 10  # Strong positive reward
        elif trend_direction < -0.3 and last_price_change_pct < 0:  # Downtrend and negative change
            reward += 10  # Strong positive reward

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if rsi_value < 30:  # Oversold
            reward += 10  # Buy signal
        elif rsi_value > 70:  # Overbought
            reward += 10  # Sell signal

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)