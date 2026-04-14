import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices

    # Feature 1: Simple Moving Average (SMA) over the last 5 days
    sma = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else closing_prices[-1]
    
    # Feature 2: Relative Strength Index (RSI)
    price_changes = np.diff(closing_prices)
    gain = np.mean(price_changes[price_changes > 0]) if np.any(price_changes > 0) else 0
    loss = -np.mean(price_changes[price_changes < 0]) if np.any(price_changes < 0) else 0
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    # Feature 3: Average True Range (ATR)
    tr1 = high_prices[1:] - low_prices[1:]  # Current high - current low
    tr2 = np.abs(high_prices[1:] - closing_prices[:-1])  # Current high - previous close
    tr3 = np.abs(low_prices[1:] - closing_prices[:-1])  # Current low - previous close
    atr = np.mean(np.maximum(tr1, np.maximum(tr2, tr3))) if len(tr1) > 0 else 0
    
    features = [sma, rsi, atr]
    return np.array(features)

def intrinsic_reward(enhanced_state):
    # enhanced_state[0:120] = raw state
    # enhanced_state[120:123] = regime_vector
    # enhanced_state[123:] = your features
    regime = enhanced_state[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # STRONG NEGATIVE reward for BUY-aligned features
        # Assume features related to buying are present; otherwise, we can adjust
    elif risk_level > 0.4:
        reward += -20  # moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features
      
    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(reward)