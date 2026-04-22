import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    trading_volumes = s[4::6]  # Extract trading volumes

    # Feature 1: Exponential Moving Average (EMA) - 10 days
    ema_period = 10
    ema = np.zeros(len(closing_prices))
    ema[ema_period - 1] = np.mean(closing_prices[:ema_period])
    for i in range(ema_period, len(closing_prices)):
        ema[i] = (closing_prices[i] * (2 / (ema_period + 1))) + (ema[i - 1] * (1 - (2 / (ema_period + 1))))

    # Feature 2: Relative Strength Index (RSI) - 14 days
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: Bollinger Bands (20 periods)
    rolling_mean = np.convolve(closing_prices, np.ones(20) / 20, mode='valid')
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    # Feature 4: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * trading_volumes) / np.sum(trading_volumes) if np.sum(trading_volumes) != 0 else 0

    # Compile features
    features = [ema[-1], rsi, upper_band[-1] if len(upper_band) > 0 else np.nan,
                lower_band[-1] if len(lower_band) > 0 else np.nan, vwap]

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(features) if np.std(features) > 0 else 1  # Avoid division by zero
    bullish_threshold = np.mean(features) + historical_std
    bearish_threshold = np.mean(features) - historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[1] > bullish_threshold:  # If RSI indicates overbought
            reward -= 50  # Strong negative for BUY-aligned features
        else:
            reward += 10  # Mild positive for SELL features
    elif risk_level > 0.4:
        if features[1] > bullish_threshold:  # If RSI indicates overbought
            reward -= 20  # Moderate negative for BUY signals
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 20 * features[0]  # Positive reward aligned with EMA
        else:
            reward -= 10 * features[0]  # Negative reward aligned with EMA
            
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < bearish_threshold:  # If RSI indicates oversold
            reward += 15  # Positive reward for buying in oversold condition
        elif features[1] > bullish_threshold:  # If RSI indicates overbought
            reward -= 15  # Negative reward for buying in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range