import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV)
    closing_prices = s[0:120:6]  # Extracting closing prices
    volumes = s[4:120:6]          # Extracting trading volumes

    # Feature 1: Price Momentum (percentage change over the last 3 days)
    price_momentum = (closing_prices[-1] - closing_prices[-4]) / closing_prices[-4] if closing_prices[-4] != 0 else 0

    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0

    # Feature 3: Relative Strength Index (RSI) over the last 14 days (or fewer if not enough data)
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = calculate_rsi(closing_prices[-14:])  # Calculate RSI for the last 14 days

    features = [price_momentum, avg_volume, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 - Risk Management
    if risk_level > 0.7:
        # Strong negative reward for BUY signals
        reward = -40  # Example strong penalty
        # If SELL aligned features, give a mild positive reward
        reward += 5  # Example mild reward for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward = -20  # Example moderate penalty

    # Priority 2 - Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Reward for bullish trend features
        elif trend_direction < -0.3:
            reward += 10  # Reward for bearish trend features

    # Priority 3 - Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward for mean-reversion features

    # Priority 4 - High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)