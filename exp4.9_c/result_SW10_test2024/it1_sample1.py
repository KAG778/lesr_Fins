import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract volumes
    features = []

    # 1. Daily Price Change (percentage)
    price_changes = np.diff(closing_prices) / closing_prices[:-1]  # Percentage change
    features.append(price_changes[-1] if len(price_changes) > 0 else 0)

    # 2. 10-Day Moving Average
    if len(closing_prices) >= 10:
        moving_average = np.mean(closing_prices[-10:])  # Last 10 days
    else:
        moving_average = closing_prices[-1]  # Fallback
    features.append(moving_average)

    # 3. RSI (14-day)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        average_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        average_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        if average_loss == 0:
            return 100  # Avoid division by zero
        rs = average_gain / average_loss
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)
    features.append(rsi)

    # 4. Volatility (historical standard deviation over the last 20 days)
    if len(closing_prices) >= 20:
        volatility = np.std(closing_prices[-20:])  # Standard deviation
    else:
        volatility = np.std(closing_prices)  # Fallback
    features.append(volatility)

    # 5. Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0
    features.append(vwap)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Compute relative thresholds for dynamic reward mechanism
    average_volatility = np.mean(enhanced_s[123:][3:])  # Assume volatility is at index 3
    high_vol_threshold = average_volatility + 1.5 * np.std(enhanced_s[123:][3:])  # 1.5 std dev above mean
    low_vol_threshold = average_volatility - 1.5 * np.std(enhanced_s[123:][3:])  # 1.5 std dev below mean
    
    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward += -40  # Strong negative for BUY
        reward += +10   # Mild positive for SELL
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Strong positive for upward features
        else:  # Downtrend
            reward += 20  # Strong positive for downward features

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3:
        if risk_level < 0.3:  # Low risk
            reward += 10  # Reward mean-reversion features
        else:
            reward += -5  # Penalize breakout-chasing features

    # Priority 4: High Volatility
    if volatility_level > high_vol_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%
    elif volatility_level < low_vol_threshold:
        reward *= 2.0  # Potentially increase reward if volatility is low

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds