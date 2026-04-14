import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes
    
    # Feature 1: 14-day Relative Strength Index (RSI)
    rsi_period = 14
    if len(closing_prices) >= rsi_period:
        price_changes = np.diff(closing_prices[-rsi_period:])
        gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
        loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI when insufficient data
    features.append(rsi)

    # Feature 2: 5-day Moving Average Convergence Divergence (MACD)
    if len(closing_prices) >= 26:  # Ensure we have enough data for MACD
        short_ema = np.mean(closing_prices[-12:])  # Short-term EMA (12 days)
        long_ema = np.mean(closing_prices[-26:])   # Long-term EMA (26 days)
        macd = short_ema - long_ema
    else:
        macd = 0
    features.append(macd)

    # Feature 3: Market Volatility (Standard Deviation of Returns over last 14 days)
    log_returns = np.diff(np.log(closing_prices)) if len(closing_prices) > 1 else np.array([0])
    volatility = np.std(log_returns[-14:]) if len(log_returns) >= 14 else 0
    features.append(volatility)

    # Feature 4: Recent Volume Spike Indicator (current volume vs. average)
    avg_volume = np.mean(volumes[-14:]) if len(volumes) >= 14 else 0
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_spike)

    # Feature 5: Crisis Signal (1 if volatility spikes beyond a threshold, else 0)
    crisis_signal = 1 if volatility > np.percentile(volatility, 75) else 0  # High volatility threshold
    features.append(crisis_signal)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds for risk levels based on historical data
    risk_threshold_high = 0.7  # Replace with historical std analysis if available
    risk_threshold_medium = 0.4  # Replace with historical std analysis if available
    trend_threshold = 0.3  # Example threshold, should be based on historical data

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative for BUY-aligned features
        reward += 5    # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Positive reward proportional to trend direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(max(-100, min(100, reward)))