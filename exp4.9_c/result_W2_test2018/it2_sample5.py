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

    # Feature 1: 14-day Relative Strength Index (RSI) calculation
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Average gain
    loss = np.where(delta < 0, -delta, 0).mean()  # Average loss
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    # Feature 2: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    high_prices[1:] - closing_prices[:-1], 
                    closing_prices[:-1] - low_prices[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 3: Moving Average Convergence Divergence (MACD)
    short_ema = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0
    long_ema = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0
    macd = short_ema - long_ema

    # Feature 4: Bollinger Bands Width (to gauge volatility)
    rolling_mean = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    rolling_std = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bollinger_width = (rolling_std / rolling_mean) if rolling_mean != 0 else 0

    # Feature 5: Volume Change (current volume / mean volume over last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1  # Avoid division by zero
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    # Combine features into a single array and return
    features = [rsi, atr, macd, bollinger_width, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Use historical standard deviation for dynamic thresholding
    historical_std = np.std(enhanced_s[0:120])  # Use raw state for historical volatility assessment
    high_risk_threshold = 0.7 * historical_std
    medium_risk_threshold = 0.4 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        reward -= 50  # STRONG NEGATIVE reward for risky BUY-aligned features
        reward += 10   # MILD POSITIVE reward for SELL-aligned features
        return np.clip(reward, -100, 100)  # Early exit
    elif risk_level > medium_risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level <= medium_risk_threshold:
        momentum_feature = enhanced_s[123]  # Assuming first feature is momentum
        if trend_direction > 0 and momentum_feature > 0:
            reward += 20  # Positive reward for upward trend and BUY
        elif trend_direction < 0 and momentum_feature < 0:
            reward += 20  # Positive reward for downward trend and SELL

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std and risk_level < medium_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]