import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0:120:6]  # Last 20 days of closing prices
    volumes = s[4:120:6]          # Last 20 days of volumes

    # Feature 1: Current vs Historical Volatility (volatility percentile)
    historical_volatility = np.std(closing_prices)  # Historical standard deviation
    current_volatility = np.std(np.diff(closing_prices[-5:]))  # Last 5 days volatility
    volatility_percentile = (current_volatility / (historical_volatility if historical_volatility != 0 else 1))
    features.append(volatility_percentile)

    # Feature 2: Relative Strength Index (RSI) with a longer window for better trend detection
    rsi_period = 14  # Standard RSI period
    if len(closing_prices) >= rsi_period:
        price_changes = np.diff(closing_prices[-rsi_period:])
        gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
        loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI when insufficient data
    features.append(rsi)

    # Feature 3: Maximum Drawdown over the last 20 days
    peak = np.max(closing_prices)
    drawdown = (peak - closing_prices[-1]) / peak if peak != 0 else 0
    features.append(drawdown)

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
        reward += -40  # Strong negative for BUY-aligned features under high risk
        reward += 5    # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 20 * trend_direction  # Positive reward proportional to trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    reward = max(-100, min(100, reward))

    return reward