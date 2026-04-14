import numpy as np

def revise_state(s):
    features = []

    # Closing prices
    closes = s[0::6]  # closing prices
    # Daily returns
    daily_returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0])

    # Feature 1: Mean Daily Return
    mean_daily_return = np.mean(daily_returns)
    features.append(mean_daily_return)

    # Feature 2: Volatility (Standard Deviation of Returns)
    volatility = np.std(daily_returns)
    features.append(volatility)

    # Feature 3: Relative Strength Index (RSI) for the last 14 days
    window_length = 14
    if len(daily_returns) >= window_length:
        gains = np.where(daily_returns > 0, daily_returns, 0)
        losses = np.where(daily_returns < 0, -daily_returns, 0)

        avg_gain = np.mean(gains[-window_length:])
        avg_loss = np.mean(losses[-window_length:])
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi)
    else:
        features.append(50)  # Neutral RSI

    # Feature 4: Price Momentum (current closing price vs previous closing price)
    momentum_feature = (closes[-1] - closes[-2]) / (closes[-2] if closes[-2] != 0 else 1)
    features.append(momentum_feature)

    # Feature 5: Average True Range (ATR) for volatility context
    true_ranges = np.maximum(s[2::6] - s[3::6],  # High - Low
                             np.abs(s[2::6] - s[1::6]),  # High - Previous Close
                             np.abs(s[3::6] - s[1::6]))  # Low - Previous Close
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # Last 14 days ATR
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical std
    # Use past volatility to define thresholds
    historical_std = np.std(enhanced_s[123:])  # Assuming features are in enhanced_s[123:]

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # Strong negative reward for BUY-aligned features
        reward -= 5 * historical_std  # Dynamic strong negative
        # Mild positive reward for SELL-aligned features
        reward += 0.5 * historical_std

    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 2 * historical_std  # Dynamic moderate negative

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 3 * historical_std  # Dynamic positive reward for bullish features
        elif trend_direction < -0.3:  # Downtrend
            reward += 3 * historical_std  # Dynamic positive reward for bearish features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 2 * historical_std  # Positive reward for mean-reversion alignment

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]