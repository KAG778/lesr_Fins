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

    # Feature 1: Rate of Change (5-day)
    roc = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 5 else 0
    
    # Feature 2: Exponential Moving Average (EMA) Difference (short vs. long)
    ema_short = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else 0
    ema_long = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    ema_diff = ema_short - ema_long  # EMA difference
    
    # Feature 3: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    high_prices[1:] - closing_prices[:-1], 
                    closing_prices[:-1] - low_prices[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 4: Historical Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0  # Last 20 days

    # Feature 5: Volume Change (current volume / average volume over the last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1  # Avoid division by zero
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0
    
    features = [roc, ema_diff, atr, historical_volatility, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Use historical standard deviation for relative thresholds
    historical_std = np.std(enhanced_s[123:])  # Use features for historical std calculation
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std
    volatility_threshold = 0.6 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # Strong negative reward for risky BUY-aligned features
        reward += 10   # Mild positive for SELL-aligned features
        return np.clip(reward, -100, 100)  # Early exit if in high-risk environment
    elif risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        momentum_feature = enhanced_s[123]  # Assuming first feature is momentum
        if trend_direction > 0 and momentum_feature > 0:
            reward += 20  # Positive reward for correct bullish signal
        elif trend_direction < 0 and momentum_feature < 0:
            reward += 20  # Positive reward for correct bearish signal

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]