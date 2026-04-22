import numpy as np

def revise_state(s):
    # Extract relevant OHLCV data from the raw state
    closing_prices = s[0::6]
    high_prices = s[2::6]
    low_prices = s[3::6]
    volumes = s[4::6]

    # Feature 1: Price Momentum (Percentage change from the N-day average)
    N = 5
    momentum = (closing_prices[-1] - np.mean(closing_prices[-N:])) / np.mean(closing_prices[-N:]) if len(closing_prices) >= N and np.mean(closing_prices[-N:]) != 0 else 0.0

    # Feature 2: Average True Range (ATR) to measure volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(high_prices[1:] - closing_prices[:-1], 
                                        closing_prices[:-1] - low_prices[1:]))
    atr = np.mean(true_ranges[-N:]) if len(true_ranges) >= N else 0.0

    # Feature 3: Relative Strength Index (RSI) for overbought/oversold conditions
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

    # Feature 4: Volume Change (Percentage change from the previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if len(volumes) > 1 and volumes[-2] != 0 else 0.0

    # Feature 5: Rate of Change (ROC) to measure price acceleration
    roc = (closing_prices[-1] - closing_prices[-N]) / closing_prices[-N] if len(closing_prices) >= N and closing_prices[-N] != 0 else 0.0

    # Return the new features as a numpy array
    features = [momentum, atr, rsi, volume_change, roc]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(features)
    risk_threshold = 0.7 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward += -50  # Strong negative for BUY signals
        if features[0] < 0:  # If momentum is negative (indicating a sell signal)
            reward += 10  # Mild positive for SELL signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4 * historical_std:
        if trend_direction > 0.3 and features[0] > 0:  # Positive trend and positive momentum
            reward += 20  # Strong positive reward
        elif trend_direction < -0.3 and features[0] < 0:  # Negative trend and negative momentum
            reward += 20  # Strong positive reward

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3 * historical_std:
        if features[2] > 70:  # Overbought condition
            reward += 15  # Reward for sell signal
        elif features[2] < 30:  # Oversold condition
            reward += 15  # Reward for buy signal

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]