import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state (OHLCV data).
    
    s: 120-dimensional raw state
    Returns: 1D numpy array of computed features
    """
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Volume Change (percentage change from previous day)
    volume_change = (volumes[-1] - volumes[-2]) / volumes[-2] if len(volumes) > 1 and volumes[-2] != 0 else 0

    # Feature 3: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Average gain
    loss = np.where(delta < 0, -delta, 0).mean()  # Average loss

    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI calculation

    # Feature 4: Historical Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns[-20:])  # Volatility over the last 20 days

    # New Feature 5: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    macd = short_ema - long_ema

    # Combine features into a single array and return
    features = [price_momentum, volume_change, rsi, historical_volatility, macd]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    """
    Computes intrinsic reward based on the enhanced state.
    
    enhanced_state[0:120] = raw state
    enhanced_state[120:123] = regime_vector
    enhanced_state[123:] = features
    Returns: reward value in [-100, 100]
    """
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Using features for dynamic thresholding
    risk_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for BUY-aligned features in high-risk conditions
    else:
        reward += 10  # Mild positive for SELL features in high-risk conditions

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level <= risk_threshold:
        if trend_direction > 0 and enhanced_s[123] > 0:  # Positive alignment
            reward += 20  # Positive reward for correct bullish signal
        elif trend_direction < 0 and enhanced_s[123] < 0:  # Negative alignment
            reward += 20  # Positive reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features
        reward -= 5   # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]