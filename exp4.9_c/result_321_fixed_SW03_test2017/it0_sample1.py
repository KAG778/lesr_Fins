import numpy as np

def revise_state(s):
    # s: 120d raw state
    # Return ONLY new features (NOT including s or regime)
    closing_prices = s[0::6]  # Extract closing prices from raw state
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Simple Moving Average (SMA) over the last 5 days
    sma = np.mean(closing_prices[-5:]) if len(closing_prices[-5:]) == 5 else 0

    # Feature 2: Price Change over the last two days
    price_change = closing_prices[-1] - closing_prices[-2] if len(closing_prices) > 1 else 0

    # Feature 3: Relative Strength Index (RSI) calculation
    gains = []
    losses = []
    for i in range(1, len(closing_prices)):
        change = closing_prices[i] - closing_prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)

    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0

    rs = avg_gain / avg_loss if avg_loss > 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs)) if avg_gain + avg_loss > 0 else 0

    # Return the computed features as a numpy array
    features = [sma, price_change, rsi]
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

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if len(features) > 0:
            reward += trend_direction * features[0] * 10.0  # Use SMA for reward

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 0:  # Price has decreased
            reward += 5.0  # Mild positive for mean-reversion buy
        else:  # Price has increased
            reward -= 5.0  # Penalize for chasing breakouts

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))