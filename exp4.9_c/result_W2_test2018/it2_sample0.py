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

    # Feature 2: Stochastic Oscillator
    if len(closing_prices) >= 14:
        low_min = np.min(low_prices[-14:])
        high_max = np.max(high_prices[-14:])
        stochastic = ((closing_prices[-1] - low_min) / (high_max - low_min)) * 100 if high_max > low_min else 0
    else:
        stochastic = 0.0
        
    # Feature 3: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    high_prices[1:] - closing_prices[:-1], 
                    closing_prices[:-1] - low_prices[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 4: Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    ema_long = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    ema_trend = ema_short - ema_long  # Difference for trend strength

    # Feature 5: Volume Change (current volume / mean volume over last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1  # To avoid division by zero
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    features = [roc, stochastic, atr, ema_trend, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Use historical standard deviation for dynamic thresholds
    historical_std = np.std(enhanced_s[123:])  # Features
    high_risk_threshold = 0.7 * historical_std
    low_risk_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # STRONG NEGATIVE reward for risky BUY-aligned features
        reward += 10   # MILD POSITIVE reward for SELL-aligned features
        return np.clip(reward, -100, 100)  # Early exit if in high-risk environment
    elif risk_level > low_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level <= low_risk_threshold:
        momentum_feature = enhanced_s[123]  # Assuming first feature is momentum
        if trend_direction > 0 and momentum_feature > 0:  # Aligning with upward trend
            reward += 20
        elif trend_direction < 0 and momentum_feature < 0:  # Aligning with downward trend
            reward += 20

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]