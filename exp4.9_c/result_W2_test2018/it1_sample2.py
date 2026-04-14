import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state (OHLCV data).
    
    s: 120-dimensional raw state
    Returns: 1D numpy array of computed features
    """
    closing_prices = s[0::6]  # Extract closing prices
    high_prices = s[2::6]     # Extract high prices
    low_prices = s[3::6]      # Extract low prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Percentage Price Oscillator (PPO)
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    ppo = (short_ema - long_ema) / long_ema if long_ema != 0 else 0

    # Feature 3: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], high_prices[1:] - closing_prices[:-1], closing_prices[:-1] - low_prices[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 1e-10  # Avoid division by zero
    
    # Feature 4: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0

    # Feature 5: Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Average gain
    loss = np.where(delta < 0, -delta, 0).mean()  # Average loss
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    features = [price_momentum, ppo, atr, vwap, rsi]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    """
    Computes intrinsic reward based on the enhanced state.
    
    enhanced_s[0:120] = raw state
    enhanced_s[120:123] = regime_vector
    enhanced_s[123:] = features
    Returns: reward value in [-100, 100]
    """
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0

    # Determine thresholds using historical std deviation
    historical_std = np.std(enhanced_s[123:])
    risk_threshold_high = 0.7 * historical_std
    risk_threshold_medium = 0.4 * historical_std
    volatility_threshold = 0.6 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # STRONG NEGATIVE for BUY-aligned features
        reward += 10   # MILD POSITIVE for SELL-aligned features (slight return for selling)
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > risk_threshold_medium:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        if trend_direction > 0 and enhanced_s[123] > 0:  # Positive momentum in uptrend
            reward += 20  # Reward upward features (momentum)
        elif trend_direction < 0 and enhanced_s[123] < 0:  # Negative momentum in downtrend
            reward += 20  # Reward downward features (momentum)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]