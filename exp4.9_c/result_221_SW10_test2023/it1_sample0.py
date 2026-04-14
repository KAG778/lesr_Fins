import numpy as np

def revise_state(s):
    # s: 120d raw state
    features = []

    # Extract closing prices
    closing_prices = s[0::6]  # Extract closing prices
    num_prices = len(closing_prices)

    # Feature 1: Price Momentum (current close - close n days ago)
    n_days = 5
    if num_prices > n_days:
        price_momentum = (closing_prices[-1] - closing_prices[-n_days]) / closing_prices[-n_days]
    else:
        price_momentum = 0  # Handle edge case
    features.append(price_momentum)

    # Feature 2: Volatility (standard deviation of returns)
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    if len(daily_returns) > 0:
        volatility = np.std(daily_returns)
    else:
        volatility = 0  # Handle edge case
    features.append(volatility)

    # Feature 3: Relative Strength Index (RSI)
    if num_prices >= 14:  # Ensure there are enough days for RSI calculation
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-14:]) if np.sum(gain[-14:]) != 0 else 0
        avg_loss = np.mean(loss[-14:]) if np.sum(loss[-14:]) != 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 0  # Not enough data for RSI
    features.append(rsi)

    # Feature 4: Volume Change (% change from previous day)
    volumes = s[4::6]  # Extract trading volumes
    if len(volumes) > 1 and volumes[-2] > 0:
        volume_change = (volumes[-1] - volumes[-2]) / volumes[-2]
    else:
        volume_change = 0  # Handle edge case
    features.append(volume_change)

    # Feature 5: Average True Range (ATR)
    highs = s[2::6]
    lows = s[3::6]
    true_ranges = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming positive momentum indicates a buy signal
            reward -= np.random.uniform(40, 60)  # Strong negative reward for BUY
        elif features[0] < 0:  # Assuming negative momentum indicates a sell signal
            reward += np.random.uniform(5, 15)  # Mild positive reward for SELL

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and positive momentum
            reward += np.random.uniform(10, 30)  # Positive reward for upward trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and negative momentum
            reward += np.random.uniform(10, 30)  # Positive reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 30:  # Assuming RSI < 30 is oversold
            reward += np.random.uniform(5, 15)  # Reward for buying in oversold conditions
        elif features[2] > 70:  # Assuming RSI > 70 is overbought
            reward += np.random.uniform(5, 15)  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within bounds