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

    # Feature 1: Rate of Change (ROC) to measure momentum
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Exponential Moving Averages (EMA) for trend detection
    ema_short = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    ema_long = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    ema_trend = ema_short - ema_long  # EMA difference for trend strength

    # Feature 3: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    high_prices[1:] - closing_prices[:-1], 
                    closing_prices[:-1] - low_prices[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 1e-10  # Avoid division by zero

    # Feature 4: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    macd = short_ema - long_ema

    # Feature 5: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0

    # Combine features into a single array and return
    features = [roc, ema_trend, atr, macd, vwap]
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

    # Calculate relative thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Using features for dynamic thresholding
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # STRONG NEGATIVE for risky BUY-aligned features
        reward += 10   # MILD POSITIVE for SELL-aligned features
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level <= low_risk_threshold:
        momentum_feature = enhanced_s[123]  # Assuming first feature is momentum
        if trend_direction > 0 and momentum_feature > 0:
            reward += 20  # Positive reward for alignment with upward trend
        elif trend_direction < 0 and momentum_feature < 0:
            reward += 20  # Positive reward for alignment with downward trend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < low_risk_threshold:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]