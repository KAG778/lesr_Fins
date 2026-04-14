import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    num_days = len(closing_prices)

    # Feature 1: 14-Day Exponential Moving Average (EMA)
    if num_days >= 14:
        ema = np.mean(closing_prices[-14:])  # Simple EMA for simplicity
    else:
        ema = closing_prices[-1] if num_days > 0 else 0.0
    
    # Feature 2: 14-Day Average True Range (ATR)
    true_ranges = np.abs(np.diff(closing_prices))  # True Range calculated as abs(diff)
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.nan

    # Feature 3: Bollinger Bands Width (20-Day)
    if num_days >= 20:
        rolling_mean = np.mean(closing_prices[-20:])
        rolling_std = np.std(closing_prices[-20:])
        bollinger_band_width = 2 * rolling_std  # Width of the Bollinger Bands
    else:
        bollinger_band_width = 0.0

    # Feature 4: Rate of Change (ROC) for momentum
    roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if num_days > 1 else 0.0

    features = [ema, atr, bollinger_band_width, roc]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY in high-risk conditions
        reward += 5.0 * features[0]  # Mild positive for SELL-aligned features via EMA
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if features[3] > 0:  # Positive momentum via ROC
            reward += 10.0 * features[3]  # Positive reward for momentum alignment
        else:  # Negative momentum
            reward += 10.0 * -features[3]  # Positive reward for negative momentum

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 0.5 * np.nanmean(features[1:]):  # Low volatility condition
            reward += 10.0  # Positive for buying in low volatility (oversold condition)
        elif features[1] > 1.5 * np.nanmean(features[1:]):  # High volatility condition
            reward -= 10.0  # Negative for selling in high volatility (overbought condition)

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))