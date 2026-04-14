import numpy as np

def revise_state(s):
    # s: 120-dimensional raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    momentum_period = 5
    sma_period = 14

    # Feature 1: Adaptive Relative Strength Index (RSI)
    price_changes = np.diff(closing_prices)
    avg_gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
    avg_loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Calculate dynamic thresholds for RSI
    historical_std = np.std(closing_prices)
    rsi_lower_threshold = 30 - (historical_std / 10)  # Example of dynamic adjustment
    rsi_upper_threshold = 70 + (historical_std / 10)

    # Feature 2: Bollinger Bands
    sma = np.mean(closing_prices[-sma_period:])  # Last SMA value
    std_dev = np.std(closing_prices[-sma_period:])  # Last 14 days std deviation
    upper_band = sma + (2 * std_dev)
    lower_band = sma - (2 * std_dev)

    # Feature 3: Average True Range (ATR)
    high_prices = s[2:120:6]
    low_prices = s[3:120:6]
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    high_prices[1:] - closing_prices[:-1],
                    closing_prices[:-1] - low_prices[1:])
    atr = np.mean(tr[-14:]) if len(tr) > 0 else 0  # ATR over the last 14 days

    features = [rsi, upper_band, lower_band, atr]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Get new features
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY
        reward += np.random.uniform(5, 10)   # Mild positive for SELL
    elif risk_level > 0.4:
        reward -= np.random.uniform(5, 20)    # Moderate negative for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        # Momentum alignment rewards
        if trend_direction > 0 and features[0] < 70:  # If in uptrend and not overbought
            reward += 20
        elif trend_direction < 0 and features[0] > 30:  # If in downtrend and not oversold
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 30:  # RSI indicates oversold
            reward += 10
        elif features[0] > 70:  # RSI indicates overbought
            reward -= 10

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward stays within bounds