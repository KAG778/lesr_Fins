import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    closing_prices = s[0::6]  # Extract closing prices
    features = []

    # 1. Rate of Change (ROC) - Sensitivity to trends
    try:
        roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]  # Change from last day to the day before
    except ZeroDivisionError:
        roc = 0.0
    features.append(roc)

    # 2. Relative Strength Index (RSI) - Measure of momentum
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Mean gain
    loss = np.abs(np.where(delta < 0, delta, 0)).mean()  # Mean loss
    try:
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
    except ZeroDivisionError:
        rsi = 50  # Neutral RSI
    features.append(rsi)

    # 3. Volatility (standard deviation of closing prices over the last 20 days)
    volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(volatility)

    # 4. Moving Average Convergence Divergence (MACD) - Trend-following indicator
    short_term_ema = np.mean(closing_prices[-12:])  # Short-term EMA (last 12 days)
    long_term_ema = np.mean(closing_prices[-26:])  # Long-term EMA (last 26 days)
    macd = short_term_ema - long_term_ema
    features.append(macd)

    # 5. Average True Range (ATR) - Measure of market volatility
    true_ranges = np.maximum(closing_prices[1:] - closing_prices[:-1], 
                             np.abs(closing_prices[1:] - closing_prices[:-1]))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 15 else 0.0  # ATR over 14 days
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Set dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Considering features for dynamic thresholds
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std
    volatility_threshold_high = 0.6 * historical_std

    # Priority 1: Risk Management
    if risk_level > risk_threshold_high:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY actions
    elif risk_level > risk_threshold_moderate:
        reward -= np.random.uniform(10, 20)  # MODERATE NEGATIVE for BUY actions

    # Priority 2: Trend Following (when risk is low)
    if abs(trend_direction) > 0.3 and risk_level <= risk_threshold_moderate:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Reward for BUY signals in an uptrend
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Reward for SELL signals in a downtrend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        # Reward mean-reversion (oversold→buy, overbought→sell) based on RSI
        if enhanced_s[123] < 30:  # Oversold
            reward += 15.0  # Reward for buying an oversold signal
        elif enhanced_s[123] > 70:  # Overbought
            reward += 15.0  # Reward for selling an overbought signal

    # Priority 4: High Volatility
    if volatility_level > volatility_threshold_high and risk_level <= risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50% during high volatility

    return float(np.clip(reward, -100, 100))  # Clamp the reward between -100 and 100