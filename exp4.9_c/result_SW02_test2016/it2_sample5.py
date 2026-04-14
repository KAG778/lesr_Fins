import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract volumes
    
    if len(closing_prices) < 20 or len(volumes) < 20:
        return np.zeros(5)  # Not enough data to calculate features

    # Feature 1: Average True Range (ATR) for volatility measurement
    high_prices = s[2::6]  # Extract high prices
    low_prices = s[3::6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:])  # ATR over the last 14 days

    # Feature 2: Bollinger Bands Distance
    sma = np.mean(closing_prices[-20:])  # 20-day SMA
    std_dev = np.std(closing_prices[-20:])  # 20-day standard deviation
    price_distance_to_band = (closing_prices[-1] - sma) / std_dev if std_dev != 0 else 0

    # Feature 3: Volatility-Adjusted Momentum
    momentum = closing_prices[-1] - closing_prices[-2]
    historical_volatility = np.std(closing_prices[-14:]) if len(closing_prices) >= 14 else 1  # Avoid division by zero
    volatility_adjusted_momentum = momentum / historical_volatility

    # Feature 4: Volume Change Percentage
    recent_volume = volumes[-1]
    previous_volume = volumes[-2]
    volume_change_percentage = ((recent_volume - previous_volume) / previous_volume) * 100 if previous_volume != 0 else 0

    # Feature 5: RSI Variation
    gains = np.where(np.diff(closing_prices[-14:]) > 0, np.diff(closing_prices[-14:]), 0)
    losses = -np.where(np.diff(closing_prices[-14:]) < 0, np.diff(closing_prices[-14:]), 0)
    average_gain = np.mean(gains) if len(gains) > 0 else 0
    average_loss = np.mean(losses) if len(losses) > 0 else 0
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))  # Standard RSI calculation

    features = [atr, price_distance_to_band, volatility_adjusted_momentum, volume_change_percentage, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds for risk level using historical data
    historical_std = np.std(enhanced_s[123:])  # Assume features are in the context of risk
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_moderate = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY signals
        reward += 10   # Mild positive reward for SELL signals
    elif risk_level > risk_threshold_moderate:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_moderate:
        reward += 30 if trend_direction > 0 else 20  # Positive reward for following the trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < risk_threshold_moderate:
        reward += 15  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_moderate:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range