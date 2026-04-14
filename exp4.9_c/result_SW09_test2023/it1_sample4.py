import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices, volumes, and calculate log returns
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]          # Trading volumes

    # Feature 1: 14-day Relative Strength Index (RSI)
    price_changes = np.diff(closing_prices)
    gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
    loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 2: 5-day Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:])  # Short-term EMA (12 days)
    long_ema = np.mean(closing_prices[-26:])   # Long-term EMA (26 days)
    macd = short_ema - long_ema
    features.append(macd)

    # Feature 3: Market Volatility (Standard Deviation of Returns over last 14 days)
    log_returns = np.diff(np.log(closing_prices))
    volatility = np.std(log_returns[-14:]) if len(log_returns) >= 14 else 0
    features.append(volatility)

    # Feature 4: Recent Volume Spike Indicator (current volume vs. average)
    avg_volume = np.mean(volumes[-14:]) if len(volumes) >= 14 else 0
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    features.append(volume_spike)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk_level
    risk_threshold_high = 0.7  # Placeholder for historical calculation
    risk_threshold_medium = 0.4  # Placeholder for historical calculation

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward += -40  # STRONG NEGATIVE for BUY-aligned features
        reward += 10   # MILD POSITIVE for SELL-aligned features
    elif risk_level > risk_threshold_medium:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Strong reward for upward momentum alignment
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Strong reward for downward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within [-100, 100]
    return float(max(-100, min(100, reward)))