import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices
    trading_volumes = s[4:120:6]  # Extract trading volumes
    
    features = []

    # Feature 1: Price Change (percentage change from previous day)
    price_changes = np.diff(closing_prices, prepend=closing_prices[0]) / closing_prices * 100
    features.append(np.mean(price_changes[-5:]))  # Average price change over the last 5 days
    
    # Feature 2: Volatility (standard deviation of price changes)
    features.append(np.std(price_changes))  # Standard deviation of price changes

    # Feature 3: Relative Strength Index (RSI)
    rsi_period = 14
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    avg_gain = np.mean(gains[-rsi_period:]) if len(gains) >= rsi_period else 0
    avg_loss = np.mean(losses[-rsi_period:]) if len(losses) >= rsi_period else 0
    rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)  # Adding the latest RSI value

    # Feature 4: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    macd = short_ema - long_ema
    features.append(macd)  # Latest MACD value

    # Feature 5: Volume Change (percentage change from previous day)
    volume_changes = np.diff(trading_volumes, prepend=trading_volumes[0]) / trading_volumes * 100
    features.append(np.mean(volume_changes[-5:]))  # Average volume change over the last 5 days
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # New features from revise_state
    reward = 0.0

    # Calculate relative thresholds based on historical standard deviation
    price_volatility = features[1]  # Standard deviation of price changes
    mean_price_change = features[0]  # Average price change over the last 5 days
    mean_volume_change = features[4]  # Average volume change over the last 5 days

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.clip(50 * mean_price_change, 20, 50)  # Strong negative for BUY-aligned features
        reward += np.clip(5 * mean_volume_change, 0, 5)  # Mild positive for SELL features if volume is increasing
    elif risk_level > 0.4:
        reward -= np.clip(20 * mean_price_change, 10, 20)  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += max(10 * mean_price_change, 10)  # Reward for upward trend
        else:
            reward += max(10 * -mean_price_change, 10)  # Reward for downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Assuming negative values indicate oversold condition
            reward += 15.0  # Positive reward for buying in oversold condition
        elif features[0] > 0.1:  # Assuming positive values indicate overbought condition
            reward -= 15.0  # Negative reward for buying in overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6:
        reward *= 0.5  # Reduce reward magnitude by 50% in high volatility conditions

    return np.clip(reward, -100, 100)  # Ensure reward is within the specified range