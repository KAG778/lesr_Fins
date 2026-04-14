import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days OHLCV)
    closing_prices = s[0:120:6]  # Extract closing prices
    n = len(closing_prices)
    
    # Feature 1: Bollinger Bands
    if n >= 20:
        moving_average = np.mean(closing_prices[-20:])
        std_dev = np.std(closing_prices[-20:])
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        bollinger_band_width = (upper_band - lower_band) / moving_average
    else:
        bollinger_band_width = 0

    # Feature 2: Exponential Moving Average (EMA)
    ema = np.mean(closing_prices[-10:]) if n >= 10 else 0
    
    # Feature 3: Max Drawdown (from peak to current price)
    peak = np.max(closing_prices)
    max_drawdown = (peak - closing_prices[-1]) / peak if peak != 0 else 0
    
    features = [bollinger_band_width, ema, max_drawdown]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assuming positive Bollinger Band width aligns with buying
            reward = np.random.uniform(-50, -30)  # Strong negative reward
        else:  # Selling in high risk
            reward = np.random.uniform(5, 10)  # Mild positive reward for sell-aligned features
    elif risk_level > 0.4:
        if features[0] > 0:  # Again, positive Bollinger Band width aligns with buying
            reward = np.random.uniform(-20, -10)  # Moderate negative reward

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3 and features[1] > 0:  # Uptrend & positive EMA
            reward += 10  # Positive reward
        elif trend_direction < -0.3 and features[1] < 0:  # Downtrend & negative EMA
            reward += 10  # Positive reward

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < 0.1:  # Low max drawdown indicates oversold condition
            reward += 10  # Reward for buying on oversold condition
        elif features[2] > 0.1:  # High max drawdown indicates overbought condition
            reward += 10  # Reward for selling on overbought condition

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50% for uncertain market

    # Ensure reward is within [-100, 100]
    reward = max(min(reward, 100), -100)
    
    return reward