import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    price_changes = np.diff(closing_prices) / closing_prices[:-1]  # Percentage change
    moving_average = np.convolve(closing_prices, np.ones(5)/5, mode='valid')  # 5-day moving average

    # Calculate RSI
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    avg_gain = np.mean(gains[-14:])  # Average gain over the last 14 days
    avg_loss = np.mean(losses[-14:])  # Average loss over the last 14 days
    
    # Handle edge case for division by zero
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [
        price_changes[-1] if len(price_changes) > 0 else 0,  # Latest price change
        moving_average[-1] if len(moving_average) > 0 else 0,  # Latest moving average
        rsi  # RSI value
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_state):
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward = -40  # STRONG NEGATIVE reward for risk level > 0.7
        return reward  # Immediate return, no further checks
    elif risk_level > 0.4:
        reward = -10  # Moderate negative reward for risk level > 0.4

    # Priority 2: Trend Following (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Determine if we are in an uptrend or downtrend
        if trend_direction > 0.3:  # Uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Here you might want to implement conditions based on computed features
        # For simplicity, just assume some reward for mean-reversion
        reward += 5  # Reward mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return reward