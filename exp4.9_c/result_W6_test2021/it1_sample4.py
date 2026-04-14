import numpy as np

def revise_state(s):
    # Extract relevant information from the raw state
    closing_prices = s[0:120:6]  # Closing prices for the last 20 days
    volumes = s[4:120:6]          # Volumes for the last 20 days

    # Feature 1: Price Rate of Change (ROC) over the last 14 days
    roc = (closing_prices[-1] - closing_prices[-14]) / closing_prices[-14] if closing_prices[-14] != 0 else 0

    # Feature 2: Average Volume over the last 10 days
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes) if len(volumes) > 0 else 0

    # Feature 3: Exponential Moving Average (EMA) of Closing Prices over the last 14 days
    def calculate_ema(prices, span=14):
        weights = np.exp(np.linspace(-1, 0, span))
        weights /= weights.sum()
        return np.dot(prices[-span:], weights)

    ema = calculate_ema(closing_prices)

    # Feature 4: Standard Deviation of Price Change (Volatility) over the last 20 days
    price_changes = np.diff(closing_prices)
    volatility = np.std(price_changes)

    # Return the computed features in a numpy array
    features = [roc, avg_volume, ema, volatility]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY signals in high-risk environments
        reward -= 50
        # Mild positive reward for SELL signals
        reward += 10
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 25

    # If not heavily risky, evaluate trend and volatility
    if risk_level <= 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            if trend_direction > 0:
                reward += 25  # Positive reward for upward momentum
            else:
                reward += 25  # Positive reward for downward momentum (correct bearish bet)

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Reward mean-reversion features
            reward += 15  # Reward for mean-reversion signals

        # Priority 4 — HIGH VOLATILITY
        if volatility_level > np.std(np.diff(enhanced_s[0:120:6])):  # Relative threshold based on price changes
            reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure the reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)