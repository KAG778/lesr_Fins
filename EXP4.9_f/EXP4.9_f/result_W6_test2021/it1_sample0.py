import numpy as np

def revise_state(s):
    # Extract OHLCV data from the raw state
    closing_prices = s[0::6]
    high_prices = s[2::6]
    low_prices = s[3::6]
    volumes = s[4::6]

    # Feature 1: Price Momentum (last day closing price - average of last N days closing prices)
    N = 5
    momentum = closing_prices[-1] - np.mean(closing_prices[-N:]) if len(closing_prices) >= N else 0.0

    # Feature 2: Average True Range (ATR) to measure volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], high_prices[1:] - closing_prices[:-1], closing_prices[:-1] - low_prices[1:])
    atr = np.mean(true_ranges[-N:]) if len(true_ranges) >= N else 0.0

    # Feature 3: Rate of Change (ROC) over the last N days
    roc = (closing_prices[-1] - closing_prices[-N]) / closing_prices[-N] if len(closing_prices) >= N and closing_prices[-N] != 0 else 0.0

    # Feature 4: Volume Spike (current volume vs average volume over last N days)
    avg_volume = np.mean(volumes[-N:]) if len(volumes) >= N else 0.0
    volume_spike = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0.0

    # Feature 5: RSI (Relative Strength Index) for mean reversion signals
    def compute_rsi(prices, period=14):
        if len(prices) < period:
            return 0
        gains = np.where(np.diff(prices) > 0, np.diff(prices), 0)
        losses = np.where(np.diff(prices) < 0, -np.diff(prices), 0)
        avg_gain = np.mean(gains[-period:]) if np.mean(gains[-period:]) != 0 else 0
        avg_loss = np.mean(losses[-period:]) if np.mean(losses[-period:]) != 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices[-14:])

    # Return the new features as a numpy array
    features = [momentum, atr, roc, volume_spike, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate historical thresholds based on the standard deviation
    historical_std = np.std(features)
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward += -40  # Strong negative for BUY signals
        if features[0] < 0:  # If momentum is negative (suggesting a sell)
            reward += 10  # Mild positive for SELL signals
    elif risk_level > 0.4 * historical_std:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold and features[0] > 0:  # Positive trend and positive momentum
            reward += 20  # Strong positive reward
        elif trend_direction < -trend_threshold and features[0] < 0:  # Negative trend and negative momentum
            reward += 20  # Strong positive reward

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        if features[4] > 70:  # Overbought condition
            reward += 15  # Reward for sell signal
        elif features[4] < 30:  # Oversold condition
            reward += 15  # Reward for buy signal

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]