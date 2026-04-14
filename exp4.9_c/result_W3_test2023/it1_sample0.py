import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Every 6th element starting from index 0 (closing prices)
    volumes = s[4::6]         # Every 6th element starting from index 4 (volumes)
    
    # Feature 1: Price Momentum (last day vs. average of previous 5 days)
    price_momentum = closing_prices[-1] - np.mean(closing_prices[-6:-1]) if len(closing_prices) >= 6 else 0.0
    
    # Feature 2: Standard Deviation of Closing Prices (Volatility)
    price_volatility = np.std(closing_prices) if len(closing_prices) > 1 else 0.0

    # Feature 3: Average Volume Change
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    recent_volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0.0

    # Feature 4: RSI Calculation
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = -np.where(deltas < 0, deltas, 0)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
        
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_value = calculate_rsi(closing_prices)

    features = [price_momentum, price_volatility, recent_volume_change, rsi_value]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123]) if len(enhanced_s[123:]) > 0 else 1  # Avoid division by zero
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY-aligned features
        reward += np.random.uniform(5, 10)    # Mild positive reward for SELL-aligned features

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > trend_threshold and risk_level < risk_threshold:
        if trend_direction > trend_threshold:  # Uptrend
            reward += 10  # Positive reward for upward features
        elif trend_direction < -trend_threshold:  # Downtrend
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if enhanced_s[123] < 30:  # Oversold
            reward += 15  # Reward for buying
        elif enhanced_s[123] > 70:  # Overbought
            reward -= 10  # Penalize for selling

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]