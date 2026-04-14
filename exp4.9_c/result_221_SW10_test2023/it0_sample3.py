import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[::6]  # Extract closing prices from the raw state
    num_days = len(closing_prices)
    
    # Calculate RSI
    if num_days >= 14:  # Ensure there are enough days for RSI calculation
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-14:]) if np.sum(gain[-14:]) != 0 else 0
        avg_loss = np.mean(loss[-14:]) if np.sum(loss[-14:]) != 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.nan  # Not enough data for RSI

    # Calculate MACD
    if num_days >= 26:  # Ensure there are enough days for MACD calculation
        ema12 = np.mean(closing_prices[-12:]) if len(closing_prices[-12:]) > 0 else 0
        ema26 = np.mean(closing_prices[-26:]) if len(closing_prices[-26:]) > 0 else 0
        macd = ema12 - ema26
    else:
        macd = np.nan  # Not enough data for MACD

    # Calculate Average True Range (ATR)
    highs = s[2::6]
    lows = s[3::6]
    true_ranges = np.maximum(highs[1:] - lows[1:], highs[1:] - closing_prices[:-1], closing_prices[:-1] - lows[1:])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else np.nan  # 14-day ATR
    
    # Create features list, ensuring to handle NaN cases
    features = [rsi if not np.isnan(rsi) else 0,
                macd if not np.isnan(macd) else 0,
                atr if not np.isnan(atr) else 0]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if enhanced_s[123] > 0:  # Assuming feature index 123 corresponds to a BUY signal
            reward = np.random.uniform(-50, -30)  # Strong negative reward for BUY
        else:
            reward = np.random.uniform(5, 10)  # Mild positive reward for SELL
    elif risk_level > 0.4:
        if enhanced_s[123] > 0:
            reward = np.random.uniform(-15, -5)  # Moderate negative reward for BUY
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and enhanced_s[123] > 0:  # BUY signal aligned with uptrend
            reward = np.random.uniform(10, 20)  # Positive reward
        elif trend_direction < -0.3 and enhanced_s[123] < 0:  # SELL signal aligned with downtrend
            reward = np.random.uniform(10, 20)  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if enhanced_s[123] > 0:  # Assuming feature index 123 corresponds to a BUY signal in oversold
            reward = np.random.uniform(5, 15)  # Reward for buying in oversold conditions
        elif enhanced_s[123] < 0:  # Assuming feature index 123 corresponds to a SELL signal in overbought
            reward = np.random.uniform(5, 15)  # Reward for selling in overbought conditions

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)