import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    features = []

    # Feature 1: Price Momentum (simple momentum indicator)
    price_momentum = s[6] - s[0]  # Current closing price - Closing price 20 days ago
    features.append(price_momentum)

    # Feature 2: Average Volume over the last 20 days
    avg_volume = np.mean(s[4::6])  # Average of the trading volume
    if avg_volume > 0:  # Prevent division by zero
        volume_change = (s[4] - avg_volume) / avg_volume  # Current volume vs average volume
    else:
        volume_change = 0
    features.append(volume_change)

    # Feature 3: Closing Price Relative Strength (RSI-like feature)
    close_prices = s[0::6]
    gains = np.maximum(np.diff(close_prices), 0)
    losses = -np.minimum(np.diff(close_prices), 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI calculation
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
    
    reward = 0  # Initialize reward

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        # STRONG NEGATIVE reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # Strong negative for risk
        # MILD POSITIVE reward for SELL-aligned features
        reward += np.random.uniform(5, 10)  # Milder positive for selling
    elif risk_level > 0.4:
        # Moderate negative reward for BUY signals
        reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            # Reward bullish features (e.g., momentum)
            reward += np.random.uniform(10, 25)  # Positive reward for bullish alignment
        elif trend_direction < -0.3:
            # Reward bearish features
            reward += np.random.uniform(10, 25)  # Positive reward for bearish alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Reward mean-reversion features (oversold/buy, overbought/sell)
        reward += np.random.uniform(5, 15)  # Positive reward for mean-reversion alignment
        # Penalize breakout-chasing features
        reward -= np.random.uniform(5, 15)  # Negative for chasing breakouts

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]