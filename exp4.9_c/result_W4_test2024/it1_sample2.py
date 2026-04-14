import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    num_days = len(closing_prices)
    
    features = []

    # Feature 1: Exponential Moving Average (EMA) - 10 days
    if num_days >= 10:
        ema = np.mean(closing_prices[-10:])  # Simple EMA for demonstration
    else:
        ema = np.mean(closing_prices)  # Fallback to mean if not enough data
    features.append(ema)

    # Feature 2: Average True Range (ATR) - last 14 days
    if num_days >= 14:
        high = s[1::6][1:]  # Extract high prices
        low = s[2::6][1:]   # Extract low prices
        close = closing_prices[:-1]  # Use previous day's closing price
        tr = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
        atr = np.mean(tr[-14:])
    else:
        atr = np.std(closing_prices)  # Fallback to std as a rough volatility measure
    features.append(atr)

    # Feature 3: Volume Weighted Average Price (VWAP)
    if num_days > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) else 0
    else:
        vwap = 0
    features.append(vwap)

    # Feature 4: Z-score of RSI (to measure relative strength)
    if num_days >= 15:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = np.abs(np.where(deltas < 0, deltas, 0)).mean()
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi_mean = np.mean([rsi])  # Replace with historical mean if possible
        rsi_std = np.std([rsi])  # Replace with historical std if possible
        z_score_rsi = (rsi - rsi_mean) / rsi_std if rsi_std != 0 else 0
    else:
        z_score_rsi = 0
    features.append(z_score_rsi)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Define historical thresholds for dynamic risk levels
    historical_std_threshold = np.std(features)  # You may want to replace this with historical values
    risk_threshold_high = 0.5 + historical_std_threshold
    risk_threshold_moderate = 0.5

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals
        reward += 5 * features[0]  # Mild positive for SELL signals based on EMA
    elif risk_level > risk_threshold_moderate:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0.3:  # Uptrend
            reward += 10 * features[0]  # Positive reward for favorable trend (EMA)
        elif trend_direction < -0.3:  # Downtrend
            reward += 10 * features[2]  # Positive reward for downward features (VWAP)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[3] < -1:  # Z-score RSI < -1 indicates oversold
            reward += 10  # Buy signal
        elif features[3] > 1:  # Z-score RSI > 1 indicates overbought
            reward += 10  # Sell signal
        else:
            reward -= 10  # Penalize breakout chasing

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds