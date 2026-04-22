import numpy as np

def revise_state(s):
    # s: 120d raw state (OHLCV interleaved)
    closing_prices = s[::6]  # Extracting closing prices
    trading_volumes = s[4::6]  # Extracting trading volumes

    # Feature 1: Price Change Percentage (current - previous)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0

    # Feature 2: Average Volume
    average_volume = np.mean(trading_volumes) if len(trading_volumes) > 0 else 0.0

    # Feature 3: Price Momentum (current - closing price 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) > 5 else 0.0

    # Feature 4: Exponential Moving Average (EMA) for trend detection
    ema = np.mean(closing_prices[-10:])  # Last 10 days EMA

    # Feature 5: Volatility (Standard deviation of the last 10 closing prices)
    volatility = np.std(closing_prices[-10:]) if len(closing_prices) > 10 else 0.0

    # Feature 6: Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return 50.0  # Neutral RSI if not enough data
        deltas = np.diff(prices)
        gain = np.mean(deltas[deltas > 0])  # Average gain
        loss = -np.mean(deltas[deltas < 0])  # Average loss
        rs = gain / loss if loss != 0 else 0  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(closing_prices)

    features = [price_change_pct, average_volume, price_momentum, ema, volatility, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Calculate relative thresholds based on historical data
    thresholds = {
        "high_risk": 0.7,  # Example threshold for risk
        "low_risk": 0.4,   # Example threshold for low risk
        "trend_threshold": 0.3,  # Example threshold for trend
        "volatility_threshold": 0.6  # Example threshold for high volatility
    }

    # Priority 1: RISK MANAGEMENT
    if risk_level > thresholds["high_risk"]:
        # Strong negative for BUY-aligned features
        reward -= np.random.uniform(30, 50)
        # Mild positive for SELL-aligned features
        reward += np.random.uniform(5, 10)
    elif risk_level > thresholds["low_risk"]:
        # Moderate negative for BUY signals
        reward -= np.random.uniform(10, 20)

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > thresholds["trend_threshold"] and risk_level < thresholds["low_risk"]:
        if trend_direction > 0:  # Uptrend
            reward += np.random.uniform(10, 20) if features[0] > 0 else 0  # Positive reward for correct buy signal
        elif trend_direction < 0:  # Downtrend
            reward += np.random.uniform(10, 20) if features[0] < 0 else 0  # Positive reward for correct sell signal

    # Priority 3: SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < thresholds["trend_threshold"] and risk_level < 0.3:
        if features[0] < -0.05:  # Assuming a strong oversold condition
            reward += np.random.uniform(10, 20)  # Reward for buying
        elif features[0] > 0.05:  # Assuming a strong overbought condition
            reward += np.random.uniform(10, 20)  # Reward for selling

    # Priority 4: HIGH VOLATILITY
    if volatility_level > thresholds["volatility_threshold"] and risk_level < thresholds["low_risk"]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]