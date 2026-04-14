import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    high_prices = s[2::6]      # Extract high prices
    low_prices = s[3::6]       # Extract low prices
    
    # Feature 1: Price Momentum (current price vs price 5 days ago)
    price_momentum = closing_prices[0] - closing_prices[5] if len(closing_prices) > 5 else 0.0

    # Feature 2: Volume Change (current volume vs average volume of last 5 days)
    current_volume = volumes[0]
    avg_volume = np.mean(volumes[:5]) if len(volumes) > 5 else 1.0  # Avoid division by zero
    volume_change = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0

    # Feature 3: Price Range (high - low over the last 20 days)
    price_range = np.max(high_prices[-20:]) - np.min(low_prices[-20:]) if len(high_prices) > 20 else 0.0

    # Feature 4: 20-Day Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.0

    # Feature 5: Relative Strength Index (RSI)
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.0
    rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))  # Avoid division by zero

    features = [price_momentum, volume_change, price_range, volatility, rsi]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    features = enhanced_s[123:]

    reward = 0.0

    # Priority 1: Risk Management
    if risk_level > 0.7:
        reward -= 40.0  # Strong negative for BUY-aligned features
        reward += 10.0 * features[1]  # Mild positive for SELL-aligned features based on volume change
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 10.0  # Strong reward for momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < 0:  # Oversold condition
            reward += 5.0  # Positive reward for potential buy
        elif features[0] > 0:  # Overbought condition
            reward -= 5.0  # Negative penalty for potential sell

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))