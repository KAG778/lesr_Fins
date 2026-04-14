import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extracting closing prices (every 6th element starting from 0)
    
    # Feature 1: Price Change (%) from the last day to the second last day
    price_change = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if closing_prices[-2] != 0 else 0
    
    # Feature 2: Moving Average (MA) of the last 5 closing prices
    if len(closing_prices) >= 5:
        moving_average = np.mean(closing_prices[-5:])
    else:
        moving_average = np.nan  # Handle edge case
    
    # Feature 3: Relative Strength Index (RSI)
    # RSI calculation
    deltas = np.diff(closing_prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.nan
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.nan
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if not np.isnan(rs) else np.nan
    
    features = [price_change, moving_average, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_state[123:]
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        if features[0] > 0:  # Assuming positive price change is a BUY signal
            reward -= np.random.uniform(30, 50)  # Strong negative reward
        # MILD POSITIVE reward for SELL-aligned features
        if features[0] < 0:  # Assuming negative price change is a SELL signal
            reward += np.random.uniform(5, 10)  # Mild positive reward
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # Again, assuming positive price change is a BUY signal
            reward -= 10  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[0] > 0:  # Upward features
            reward += 10  # Positive reward
        elif trend_direction < -0.3 and features[0] < 0:  # Downward features
            reward += 10  # Positive reward
            
    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Assuming negative price change indicates oversold
            reward += 10  # Reward for mean-reversion buy
        elif features[0] > 0:  # Assuming positive price change indicates overbought
            reward += 10  # Reward for mean-reversion sell
    
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% (uncertain market)

    return float(reward)