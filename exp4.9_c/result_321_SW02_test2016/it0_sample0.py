import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    
    # Initialize an empty list to hold new features
    features = []
    
    # Feature 1: Price Change Percentage (from the last two days)
    # Prevent division by zero
    if s[6] != 0:  # Ensure the previous closing price is not zero
        price_change_pct = (s[0] - s[6]) / s[6]  # (current_close - previous_close) / previous_close
    else:
        price_change_pct = 0.0
    features.append(price_change_pct)
    
    # Feature 2: Moving Average of Closing Prices (last 5 days)
    # Calculate moving average to identify short-term trends
    closing_prices = s[0:120:6]  # Extract closing prices (every 6th element)
    moving_avg_5 = np.mean(closing_prices[-5:])  # Last 5 closing prices
    features.append(moving_avg_5)

    # Feature 3: Relative Strength Index (RSI) (14-day period)
    # We need at least 14 price changes to calculate RSI
    if len(closing_prices) >= 14:
        deltas = np.diff(closing_prices[-14:])  # Price changes for the last 14 days
        gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
        loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50.0  # Neutral RSI when not enough data
    features.append(rsi)
    
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
        # Strong negative reward for BUY-aligned features
        reward += -40  # Strong negative for buying in dangerous conditions
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward += -20  # Moderate penalty for buying with elevated risk
    
    # Check if risk is low for further evaluations
    if risk_level < 0.4:
        # Priority 2 — TREND FOLLOWING
        if abs(trend_direction) > 0.3:
            # Trend following rewards
            if trend_direction > 0:
                # Positive trend (momentum strategies favored)
                reward += 10  # Reward for bullish conditions
            else:
                # Negative trend (correct bearish bet)
                reward += 10  # Reward for bearish conditions

        # Priority 3 — SIDEWAYS / MEAN REVERSION
        elif abs(trend_direction) < 0.3:
            # Mean-reversion strategies
            if enhanced_s[123] < 30:  # Assuming enhanced_s[123] holds an oversold feature
                reward += 5  # Reward for potential buying opportunity
            else:
                reward += -5  # Penalize breakout-chasing features

        # Priority 4 — HIGH VOLATILITY
        if volatility_level > 0.6:
            reward *= 0.5  # Reduce reward magnitude by 50%
    
    return float(reward)