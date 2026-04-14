import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: 14-day Relative Strength Index (RSI)
    if len(closing_prices) >= 14:
        price_changes = np.diff(closing_prices[-14:])
        gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
        loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50  # Neutral RSI when insufficient data
    features.append(rsi)

    # Feature 2: 5-day Moving Average Convergence Divergence (MACD)
    if len(closing_prices) >= 26:
        short_ema = np.mean(closing_prices[-12:])  # Short-term EMA (12 days)
        long_ema = np.mean(closing_prices[-26:])   # Long-term EMA (26 days)
        macd = short_ema - long_ema
    else:
        macd = 0
    features.append(macd)

    # Feature 3: Market Volatility (Standard Deviation of Returns over last 14 days)
    log_returns = np.diff(np.log(closing_prices))
    volatility = np.std(log_returns[-14:]) if len(log_returns) >= 14 else 0
    features.append(volatility)

    # Feature 4: Recent Volume Spike Indicator (current volume vs. average)
    avg_volume = np.mean(volumes[-14:]) if len(volumes) >= 14 else 1  # Avoid division by zero
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_spike)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate dynamic thresholds based on historical data
    historical_std = 0.1  # Placeholder for historical std calculation
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 40  # Strong negative for BUY-aligned features
        reward += 5    # Mild positive for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        reward += 15 * np.sign(trend_direction)  # Reward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return max(-100, min(100, reward))