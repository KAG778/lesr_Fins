import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    num_days = len(closing_prices)
    
    # Feature 1: Price Momentum (current close - previous close)
    price_momentum = closing_prices[-1] - closing_prices[-2] if num_days > 1 else 0.0
    
    # Feature 2: 14-Day Relative Strength Index (RSI)
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        average_gain = np.mean(gains[-period:]) if len(gains) >= period else 0.0
        average_loss = np.mean(losses[-period:]) if len(losses) >= period else 0.0
        rs = average_gain / average_loss if average_loss != 0 else 0.0
        return 100 - (100 / (1 + rs))
    
    rsi = calculate_rsi(closing_prices) if num_days >= 14 else 0.0
    
    # Feature 3: 5-Day Volatility (standard deviation of the last 5 closes)
    volatility = np.std(closing_prices[-5:]) if num_days >= 5 else 0.0
    
    # Feature 4: Price Change Percentage (from previous day)
    price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] * 100 if num_days > 1 else 0.0

    # Feature 5: Average Trading Volume (last 20 periods)
    volumes = s[4:120:6]  # Extract volumes
    average_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0.0

    features = [price_momentum, rsi, volatility, price_change_pct, average_volume]
    
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
        reward -= 40.0  # Strong negative for buying in risky conditions
        reward += 5.0 * features[0]  # Mild positive for selling momentum
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for buying

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 5.0 * trend_direction * features[0]  # Reward for momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # RSI < 30: Oversold
            reward += 10.0  # Positive for buying in oversold condition
        elif features[1] > 70:  # RSI > 70: Overbought
            reward -= 10.0  # Negative for selling in overbought condition

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude

    return float(np.clip(reward, -100, 100))