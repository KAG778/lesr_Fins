import numpy as np

def revise_state(s):
    features = []
    
    # Extract relevant data from the state
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes
    
    # Feature 1: Price Momentum (percentage change)
    recent_close = closing_prices[-1]
    previous_close = closing_prices[-2] if len(closing_prices) > 1 else recent_close
    momentum = (recent_close - previous_close) / previous_close if previous_close != 0 else 0
    features.append(momentum)
    
    # Feature 2: Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)
    features.append(rsi)
    
    # Feature 3: Volatility (Rolling Standard Deviation of Returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]
    if len(returns) >= 20:
        volatility = np.std(returns[-20:])  # Standard deviation of the last 20 returns
    else:
        volatility = 0
    features.append(volatility)
    
    # Feature 4: Average Volume Change (relative change)
    if len(volumes) >= 6:
        avg_volume_current = np.mean(volumes[-5:])
        avg_volume_previous = np.mean(volumes[-10:-5]) if len(volumes) > 10 else avg_volume_current
        volume_change = (avg_volume_current - avg_volume_previous) / avg_volume_previous if avg_volume_previous != 0 else 0
    else:
        volume_change = 0
    features.append(volume_change)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]  # Extract features
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # Strong negative for high risk (BUY)
        reward += np.random.uniform(5, 10)   # Mild positive for SELL
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0 and features[0] > 0:  # Uptrend and momentum confirms
            reward += 20  # Positive reward for following the trend
        elif trend_direction < 0 and features[0] < 0:  # Downtrend and momentum confirms
            reward += 20  # Positive reward for correct bearish bet

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[1] < 30:  # Oversold condition (RSI)
            reward += 15  # Reward for buying in an oversold condition
        elif features[1] > 70:  # Overbought condition (RSI)
            reward -= 15  # Penalize for buying in an overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds