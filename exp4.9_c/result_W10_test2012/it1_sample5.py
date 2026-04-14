import numpy as np

def revise_state(s):
    features = []
    
    # Closing prices (s[i*6 + 0])
    closing_prices = s[0::6]  
    
    # Feature 1: Exponential Moving Average (EMA) over 20 days
    ema = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0

    # Feature 2: Average True Range (ATR) for volatility (14-day)
    high_prices = s[2::6]
    low_prices = s[3::6]
    true_ranges = np.maximum(high_prices[1:], closing_prices[1:] - low_prices[1:], low_prices[1:] - closing_prices[:-1])
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0

    # Feature 3: Rate of Change (ROC) for momentum (over 14 days)
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if len(closing_prices) >= 15 and closing_prices[-15] != 0 else 0

    # Feature 4: Bollinger Bands Width (for volatility indication)
    moving_avg = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    moving_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bb_width = (moving_std / moving_avg) if moving_avg != 0 else 0

    features.extend([ema, atr, roc, bb_width])
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50  # Strong negative for BUY-aligned signals
        if enhanced_s[123] <= 0:  # Assuming feature[0] is SELL-aligned
            reward += 10  # Mild positive for SELL
        return max(-100, reward)  # Ensure return is within bounds

    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 30  # Strong reward for upward momentum alignment
        else:
            reward += 30  # Strong reward for downward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        rsi = enhanced_s[123][2]  # Assuming RSI is the third feature
        if rsi < 30:  # Oversold condition
            reward += 20  # Reward for buying in an oversold market
        elif rsi > 70:  # Overbought condition
            reward += -20  # Penalize for buying in an overbought market

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return max(-100, min(100, reward))  # Ensure reward is within bounds