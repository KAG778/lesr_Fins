import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    if len(closing_prices) < 20:
        return np.zeros(5)  # Not enough data to calculate features

    # Feature 1: Average True Range (ATR)
    high_prices = s[2::6]  # High prices
    low_prices = s[3::6]   # Low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:])  # ATR over the last 14 days

    # Feature 2: Bollinger Bands Distance
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
    price_distance_to_band = (closing_prices[-1] - sma) / std_dev if std_dev != 0 else 0

    # Feature 3: Volatility Adjusted Momentum
    momentum = closing_prices[-1] - closing_prices[-2]
    historical_volatility = np.std(closing_prices[-14:]) if len(closing_prices) >= 14 else 1  # Avoid division by zero
    volatility_adjusted_momentum = momentum / historical_volatility

    # Feature 4: Exponential Moving Average (EMA) Crossover
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else np.mean(closing_prices)
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else np.mean(closing_prices)
    ema_crossover = short_ema - long_ema

    # Feature 5: Relative Strength Index (RSI)
    deltas = np.diff(closing_prices[-14:])  # Daily price changes
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if (gain + loss) != 0 else 0

    features = [atr, price_distance_to_band, volatility_adjusted_momentum, ema_crossover, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_risk = np.std(enhanced_s[123:])  # Assuming features are in the context of risk
    risk_threshold_high = 0.7 * historical_risk  # High risk threshold
    risk_threshold_moderate = 0.4 * historical_risk  # Moderate risk threshold

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative for BUY signals
        reward += 10   # Mild positive for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        if trend_direction > 0:  # Uptrend
            reward += 30  # Positive reward for buying in an uptrend
        else:  # Downtrend
            reward += 30  # Positive reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range