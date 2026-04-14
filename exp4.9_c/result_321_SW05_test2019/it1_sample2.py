import numpy as np

def revise_state(s):
    # Extract closing prices and volumes from the raw state
    closing_prices = s[0:120:6]  # Closing prices
    volumes = s[4:120:6]          # Trading volumes

    # Calculate features
    # Feature 1: Price Change Percentage
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0

    # Feature 2: Relative Volume (current volume vs. average of the last 5 days)
    avg_volume = np.mean(volumes[-5:]) if np.mean(volumes[-5:]) > 0 else 1  # Avoid division by zero
    relative_volume = (volumes[-1] - avg_volume) / avg_volume

    # Feature 3: Extreme Price Movement (5-day max drawdown)
    max_price = np.max(closing_prices[-5:])
    min_price = np.min(closing_prices[-5:])
    extreme_movement = (max_price - min_price) / min_price if min_price != 0 else 0

    # Feature 4: 14-day RSI
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_value = calculate_rsi(closing_prices)

    # Return the new features
    return np.array([price_change_pct, relative_volume, extreme_movement, rsi_value])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate historical volatility (standard deviation of price changes)
    closing_prices = enhanced_s[0:120:6]
    price_changes = np.diff(closing_prices)
    historical_volatility = np.std(price_changes)

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 50  # Strong negative reward for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Strong uptrend
            reward += 20  # Positive reward for upward momentum
        elif trend_direction < -0.3:  # Strong downtrend
            reward += 20  # Positive reward for downward momentum

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] < 30:  # Oversold condition
            reward += 15  # Reward for mean-reversion buy
        elif enhanced_s[123] > 70:  # Overbought condition
            reward += 15  # Reward for mean-reversion sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within bounds
    return np.clip(reward, -100, 100)