import numpy as np

def revise_state(s):
    # Extract closing prices
    closing_prices = s[0::6]  # Every 6th element starting from index 0
    days = len(closing_prices)
    
    # Feature 1: Price Change Percentage (1-day change)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0
    
    # Feature 2: Average True Range (ATR) for volatility measurement (14 days)
    high = s[1::6]  # Highest price for the day
    low = s[2::6]   # Lowest price for the day
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - closing_prices[1:-1]), abs(low[1:] - closing_prices[1:-1])))
    atr = np.mean(tr[-14:]) if len(tr) > 14 else 0  # Mean of true range over the last 14 days
    
    # Feature 3: Exponential Moving Average Convergence Divergence (MACD)
    short_window = 12
    long_window = 26
    signal_window = 9

    ema_short = np.zeros(days)
    ema_long = np.zeros(days)
    macd = np.zeros(days)
    
    # EMA Calculation
    for i in range(days):
        if i == 0:
            ema_short[i] = closing_prices[i]
            ema_long[i] = closing_prices[i]
        else:
            ema_short[i] = (closing_prices[i] * (2 / (short_window + 1))) + (ema_short[i - 1] * (1 - (2 / (short_window + 1))))
            ema_long[i] = (closing_prices[i] * (2 / (long_window + 1))) + (ema_long[i - 1] * (1 - (2 / (long_window + 1))))
    
    macd = ema_short - ema_long
    
    # Collect features
    features = [price_change_pct, atr, macd[-1]]  # Use the most recent values
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical volatility based on last 20 days
    historical_volatility = np.std(enhanced_s[0:120:6])  # Standard deviation of closing prices

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # Strong negative for BUY
        reward += 20  # Mild positive for SELL
        return np.clip(reward, -100, 100)  # Early exit due to high risk
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 20  # Reward for correct bullish signal
        else:  # Downtrend
            reward += 20  # Reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_volatility and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)