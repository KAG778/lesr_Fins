import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []
    
    # Feature 1: Relative Strength Index (RSI) - a momentum indicator
    # Calculate RSI for the last 14 trading days
    closing_prices = s[0:120:6]  # Extract closing prices
    if len(closing_prices) < 14:
        rsi = np.nan  # Not enough data to calculate RSI
    else:
        deltas = np.diff(closing_prices)
        gain = np.where(deltas > 0, deltas, 0).mean()
        loss = np.abs(np.where(deltas < 0, deltas, 0)).mean()
        rs = gain / loss if loss != 0 else np.nan
        rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # Feature 2: Moving Average Convergence Divergence (MACD)
    # Calculate MACD line and signal line
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else np.nan
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else np.nan
    macd = short_ema - long_ema if not np.isnan(short_ema) and not np.isnan(long_ema) else np.nan
    features.append(macd)

    # Feature 3: Price Change Percentage
    # Calculate the percentage change from the opening price of the first day to the closing price of the last day
    opening_price_first_day = s[1]  # Opening price of day 0
    closing_price_last_day = s[114]  # Closing price of day 19
    if opening_price_first_day != 0:
        price_change_pct = (closing_price_last_day - opening_price_first_day) / opening_price_first_day
    else:
        price_change_pct = np.nan
    features.append(price_change_pct)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY
        reward += np.random.uniform(5, 10)   # Mild positive reward for SELL
    elif risk_level > 0.4:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    reward = max(-100, min(reward, 100))

    return reward