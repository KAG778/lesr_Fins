import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    opening_prices = s[1::6]  # Extract opening prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes
    
    # Feature 1: Price Momentum (percentage change from previous close)
    price_momentum = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 2: Average Volume Change (percentage change from the average of last 20 days)
    avg_volume = np.mean(volumes) if len(volumes) > 0 else 0.0
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0.0
    
    # Feature 3: Normalized Price Range (high - low) / previous close
    price_range = (high_prices[-1] - low_prices[-1]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    
    # Feature 4: 14-day Relative Strength Index (RSI)
    price_changes = np.diff(closing_prices)
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    
    average_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    average_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    
    rs = average_gain / average_loss if average_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    features = [
        price_momentum,  # Feature 1
        volume_change,   # Feature 2
        price_range,     # Feature 3
        rsi              # Feature 4
    ]
    
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
        if features[0] < 0:  # If price momentum is negative, reward the SELL signal
            reward += 5.0
    elif risk_level > 0.4:
        reward -= 10.0  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += trend_direction * features[0] * 20.0  # Reward for positive momentum alignment

    # Priority 3: Sideways / Mean Reversion
    if abs(trend_direction) <= 0.3 and risk_level < 0.3:
        if features[3] < 30:  # Oversold condition (RSI)
            reward += 10.0  # Buy signal
        elif features[3] > 70:  # Overbought condition (RSI)
            reward += 10.0  # Sell signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))