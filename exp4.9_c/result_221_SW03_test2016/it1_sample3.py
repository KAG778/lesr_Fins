import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Moving Average (MA) over last 5 days
    ma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan
    
    # Feature 2: Relative Strength Index (RSI) - a momentum indicator
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0).mean() if len(deltas) > 0 else 0
    loss = -np.where(deltas < 0, deltas, 0).mean() if len(deltas) > 0 else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # Feature 3: Average True Range (ATR) for volatility
    if len(closing_prices) < 14:
        atr = np.nan  # Not enough data to calculate ATR
    else:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                   np.abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr[-14:])  # 14-day ATR

    # Feature 4: Price Change Percentage (last 5 days)
    price_change_pct = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] * 100 if len(closing_prices) >= 6 and closing_prices[-6] != 0 else 0
    
    # Combine features into a single array
    features = [ma_5, rsi, atr, price_change_pct]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate relative thresholds based on historical data
    mean_risk = 0.5  # Example mean based on historical data
    std_risk = 0.1   # Example std based on historical data
    risk_threshold_low = mean_risk - std_risk
    risk_threshold_high = mean_risk + std_risk

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY
        reward += 10   # Mild positive reward for SELL
    elif risk_level > risk_threshold_low:
        reward -= 20  # Moderate negative reward for BUY

    # Priority 2 — TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold_low:
        if trend_direction > 0.3:  # Uptrend
            reward += 20  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += 20  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_low:
        reward *= 0.5  # Reduce reward magnitude by 50%

    # Ensure reward is within the bounds of [-100, 100]
    return np.clip(reward, -100, 100)