import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Rate of Change of Closing Prices (ROC)
    roc = (closing_prices[-1] - closing_prices[-5]) / closing_prices[-5] if closing_prices[-5] != 0 else 0
    
    # Feature 2: Average True Range (ATR) for Volatility
    high_prices = s[2::6]
    low_prices = s[3::6]
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:], high_prices[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # 14-day ATR
    
    # Feature 3: Z-score of Volume Change
    volume_change = np.diff(volumes)  # Change in volume
    volume_change_z = (volume_change[-1] - np.mean(volume_change)) / np.std(volume_change) if np.std(volume_change) != 0 else 0
    
    # Feature 4: Momentum Indicator (e.g., RSI)
    gains = np.where(closing_prices[1:] > closing_prices[:-1], closing_prices[1:] - closing_prices[:-1], 0)
    losses = np.where(closing_prices[1:] < closing_prices[:-1], closing_prices[:-1] - closing_prices[1:], 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    
    features = [roc, atr, volume_change_z, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    
    # Calculate relative thresholds based on historical data (standard deviations)
    risk_threshold = 0.7 * np.std(enhanced_s[123:])  # Example for dynamic threshold, adjust as per historical data
    trend_threshold = 0.3 * np.std(enhanced_s[123:])  # Example for tracking trend
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY signals
        reward += np.random.uniform(5, 10)  # MILD POSITIVE reward for SELL signals
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 10  # Positive reward for upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 10  # Positive reward for downward momentum

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features (oversold→buy, overbought→sell)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]