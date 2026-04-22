import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Initialize an empty list to hold the computed features
    features = []
    
    # Compute features based on the OHLCV data
    closing_prices = s[0::6]  # closing prices (s[0], s[6], ..., s[114])
    opening_prices = s[1::6]  # opening prices (s[1], s[7], ..., s[115])
    high_prices = s[2::6]     # high prices (s[2], s[8], ..., s[116])
    low_prices = s[3::6]      # low prices (s[3], s[9], ..., s[117])
    
    # Feature 1: Price Change Percentage (last day compared to the previous day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / (closing_prices[-2] + 1e-10)  # Avoid division by zero
    features.append(price_change_pct)

    # Feature 2: Average True Range (ATR) over the last 5 days (measures volatility)
    tr = np.maximum(high_prices[-5:] - low_prices[-5:], 
                    np.maximum(np.abs(high_prices[-5:] - closing_prices[-6:-1]),
                               np.abs(low_prices[-5:] - closing_prices[-6:-1])))
    atr = np.mean(tr)
    features.append(atr)

    # Feature 3: Exponential Moving Average (EMA) of closing prices over the last 5 days
    if len(closing_prices) >= 5:
        ema = np.mean(closing_prices[-5:])  # Simple EMA for simplicity, can be improved
    else:
        ema = closing_prices[-1]  # Fallback to the last closing price if not enough data
    features.append(ema)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Initialize reward
    reward = 0.0
    
    # Implementing the priority chain for rewards
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= 40  # Example value in the range of [-30, -50]
        # MILD POSITIVE reward for SELL-aligned features
        reward += 7   # Example value in the range of [5, 10]
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= 20  # Example value for moderate penalty
    else:
        # Risk is acceptable, evaluate trend direction
        if abs(trend_direction) > 0.3:
            if trend_direction > 0.3:
                # Uptrend, reward upward features
                reward += 20  # Example positive reward for correctly aligned trend
            elif trend_direction < -0.3:
                # Downtrend, reward downward features
                reward += 20  # Example positive reward for correctly aligned trend
        else:
            # Sideways / Mean Reversion
            if risk_level < 0.3:
                # Reward mean-reversion features and penalize breakout-chasing features
                reward += 10  # Example positive reward for mean reversion
                reward -= 5   # Example penalty for chasing breakouts

    # High volatility adjustment
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)