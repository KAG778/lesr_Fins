import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Price Change Percentage from the last day
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Exponential Moving Average (EMA) of the last 10 closing prices
    ema = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else 0.0

    # Feature 3: Average True Range (ATR) to measure volatility
    def calculate_atr(prices, period=14):
        if len(prices) < period:
            return 0.0  # Not enough data to calculate ATR
        high = np.max(prices[-period:])
        low = np.min(prices[-period:])
        close = prices[-1]
        atr = (high - low + abs(close - high) + abs(close - low)) / 3  # Simplified ATR calculation
        return atr

    atr = calculate_atr(closing_prices)

    # Feature 4: Rate of Change (ROC) over the last 10 days
    roc = ((closing_prices[-1] - closing_prices[-11]) / closing_prices[-11]) * 100 if len(closing_prices) >= 11 else 0.0

    # Feature 5: Volume Oscillator (current volume - average volume over the last 10 days)
    avg_volume = np.mean(trading_volumes[-10:]) if len(trading_volumes) >= 10 else 0.0
    volume_oscillator = trading_volumes[-1] - avg_volume

    features = [price_change_pct, ema, atr, roc, volume_oscillator]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds based on historical data
    risk_threshold = np.std(features[0])  # Using price change percentage as a proxy for risk
    momentum_threshold = np.std(features[3])  # Using ROC for momentum threshold

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:  # High risk condition
        reward -= 50 if features[0] > 0 else 10  # Strong negative for BUY, mild positive for SELL

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[3] > 0:  # Uptrend with positive momentum
            reward += 20  # Reward for alignment with upward trend
        elif trend_direction < 0 and features[3] < 0:  # Downtrend with negative momentum
            reward += 20  # Reward for alignment with downward trend

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.05:  # Strongly oversold condition
            reward += 15  # Reward for buying in mean reversion
        elif features[0] > 0.05:  # Strongly overbought condition
            reward += 15  # Reward for selling in mean reversion

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Clip reward to be within [-100, 100]