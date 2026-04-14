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

    # Feature 2: Bollinger Bands
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
    upper_band = sma + (2 * std_dev)  # Upper Bollinger Band
    lower_band = sma - (2 * std_dev)  # Lower Bollinger Band
    price_distance_to_band = (closing_prices[-1] - sma) / std_dev if std_dev != 0 else 0

    # Feature 3: Exponential Moving Average (EMA) Divergence
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    ema_divergence = short_ema - long_ema

    # Feature 4: Price Momentum
    price_momentum = closing_prices[-1] - closing_prices[-2]

    # Feature 5: Relative Strength Index (RSI)
    deltas = np.diff(closing_prices[-14:])  # Daily price changes
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if (gain + loss) != 0 else 0

    features = [atr, price_distance_to_band, ema_divergence, price_momentum, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Standard thresholds based on historical analysis
    mean_risk = 0.5  # This would be derived from historical data
    std_risk = 0.2   # This would be derived from historical data

    # Priority 1 — RISK MANAGEMENT
    if risk_level > mean_risk + 1.5 * std_risk:  # Strong negative for BUY signals
        reward -= 50
        reward += 10 if enhanced_s[123] < 0 else -10  # Mild positive for SELL signals
    elif risk_level > mean_risk + 0.5 * std_risk:  # Moderate negative for BUY signals
        reward -= 20

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < mean_risk:
        reward += 30 if trend_direction > 0 else 20  # Positive reward for following the trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < mean_risk - 0.5 * std_risk:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < mean_risk:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range