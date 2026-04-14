import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    high_prices[1:] - closing_prices[:-1], 
                    closing_prices[:-1] - low_prices[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 3: Relative Strength Index (RSI) calculation
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Average gain
    loss = np.where(delta < 0, -delta, 0).mean()  # Average loss
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    # Feature 4: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0

    # Feature 5: Rate of Change (ROC) for momentum assessment
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 6: Exponential Moving Average (EMA) difference for trend detection
    ema_short = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    ema_long = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    ema_trend = ema_short - ema_long  # EMA difference for trend strength

    features = [price_momentum, atr, rsi, vwap, roc, ema_trend]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Using features for dynamic thresholding
    risk_threshold = 0.4 * historical_std  # Threshold for risk level
    trend_threshold = 0.3 * historical_std  # Threshold for trend detection

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:  # High risk
        reward -= 50  # Strong negative reward for BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level <= risk_threshold:
        if trend_direction > 0 and enhanced_s[123] > 0:  # Positive alignment
            reward += 20  # Positive reward for correct bullish signal
        elif trend_direction < 0 and enhanced_s[123] < 0:  # Negative alignment
            reward += 20  # Positive reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]