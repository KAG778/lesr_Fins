import numpy as np

def revise_state(s):
    # Extract closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    days = len(closing_prices)

    # Feature 1: Moving Average Convergence Divergence (MACD) - more stable indicator
    def compute_macd(prices, short_window=12, long_window=26):
        ema_short = np.zeros(days)
        ema_long = np.zeros(days)
        ema_short[:short_window] = np.mean(prices[:short_window])
        ema_long[:long_window] = np.mean(prices[:long_window])
        
        for i in range(short_window, days):
            ema_short[i] = (prices[i] * (2 / (short_window + 1))) + (ema_short[i - 1] * (1 - (2 / (short_window + 1))))
        
        for i in range(long_window, days):
            ema_long[i] = (prices[i] * (2 / (long_window + 1))) + (ema_long[i - 1] * (1 - (2 / (long_window + 1))))
        
        return ema_short - ema_long

    macd = compute_macd(closing_prices)
    latest_macd = macd[-1]

    # Feature 2: Relative Strength Index (RSI) - to gauge momentum
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)

    # Feature 3: Standard Deviation of Price - to assess volatility
    price_std = np.std(closing_prices[-20:])  # Last 20 days for volatility

    # Collect features into a numpy array
    features = [latest_macd, rsi, price_std]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds based on historical data (dummy values for illustration)
    historical_std = 0.15  # Example historical standard deviation for price movements
    high_risk_threshold = 0.5 + historical_std
    medium_risk_threshold = 0.3 + historical_std
    low_risk_threshold = 0.2 + historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 - RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features
        reward += 20  # Mild positive for SELL-aligned features
    elif risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 - TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0:
            reward += 30  # Strong reward for aligned upward momentum
        else:
            reward += 30  # Strong reward for aligned downward momentum

    # Priority 3 - SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < low_risk_threshold:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 - HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Clip the reward to ensure it remains within [-100, 100]
    return np.clip(reward, -100, 100)