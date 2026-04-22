import numpy as np

def revise_state(s):
    closing_prices = s[0::6]
    high_prices = s[2::6]
    low_prices = s[3::6]
    volumes = s[4::6]
    
    N = 5  # Lookback period for most features

    # Feature 1: Price Momentum (last day closing price - average of last N days closing prices)
    momentum = closing_prices[-1] - np.mean(closing_prices[-N:]) if len(closing_prices) >= N else 0.0

    # Feature 2: Average True Range (ATR) to measure volatility
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(high_prices[1:] - closing_prices[:-1], 
                                        closing_prices[:-1] - low_prices[1:]))
    atr = np.mean(true_ranges[-N:]) if len(true_ranges) >= N else 0.0

    # Feature 3: Moving Average Convergence Divergence (MACD) for trend following
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0.0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0.0
    macd = short_ema - long_ema

    # Feature 4: Volume Change (percentage change from the previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if len(volumes) > 1 and volumes[-2] != 0 else 0.0

    # Feature 5: Relative Strength Index (RSI) for mean reversion signals
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
    features = [momentum, atr, macd, volume_change, rsi]
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
    trend_threshold = 0.3 * historical_std

    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward += -40  # Strong negative reward for BUY signals
        if features[0] < 0:  # If momentum is negative (suggesting a sell)
            reward += 10  # Mild positive for SELL signals
    elif risk_level > 0.4 * historical_std:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold and features[0] > 0:  # Positive trend and positive momentum
            reward += 15  # Strong positive reward
        elif trend_direction < -trend_threshold and features[0] < 0:  # Negative trend and negative momentum
            reward += 15  # Strong positive reward

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        if features[4] > 70:  # Overbought condition
            reward += 10  # Reward for sell signal
        elif features[4] < 30:  # Oversold condition
            reward += 10  # Reward for buy signal

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]