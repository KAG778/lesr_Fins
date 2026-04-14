import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0:120:6]  # Extract closing prices
    days = len(closing_prices)

    # Feature 1: Exponential Moving Average (EMA) - 20 days
    if days >= 20:
        ema = np.mean(closing_prices[-20:])  # Simple EMA for demonstration purposes
    else:
        ema = np.nan
    features.append(ema)

    # Feature 2: Bollinger Bands (Upper and Lower) - 20 days
    if days >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
    else:
        upper_band, lower_band = np.nan, np.nan
    features.extend([upper_band, lower_band])

    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = -np.where(delta < 0, delta, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.nan
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.nan
    rs = avg_gain / avg_loss if avg_loss != 0 else np.nan
    rsi = 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan
    features.append(rsi)

    # Feature 4: Price Change Ratio
    price_change_ratio = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if days > 1 else np.nan
    features.append(price_change_ratio)

    # Feature 5: Average True Range (ATR)
    high_prices = s[2:120:6]  # Extract high prices
    low_prices = s[3:120:6]   # Extract low prices
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                               np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.nan
    features.append(atr)

    return np.nan_to_num(np.array(features))  # Replace NaNs with 0 to ensure usability

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical volatility
    avg_volatility = np.mean([0.3, 0.4, 0.5])  # Example historical average for volatility
    high_risk_threshold = avg_volatility + 0.3 * np.std([0.3, 0.4, 0.5])
    low_risk_threshold = avg_volatility + 0.1 * np.std([0.3, 0.4, 0.5])

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative for BUY signals in high risk
        reward += 10 * (1 - risk_level)  # Mild reward for SELL signals
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif np.abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3:  # Strong uptrend
            reward += 20  # Positive reward for aligning with upward trend
        elif trend_direction < -0.3:  # Strong downtrend
            reward += 20  # Positive reward for aligning with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif np.abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for mean-reversion strategies

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]