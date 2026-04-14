import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    features = []
    
    # Feature 1: Average True Range (ATR) for Volatility
    def compute_atr(prices, n=14):
        if len(prices) < n:
            return np.nan
        tr = np.maximum(prices[1:] - prices[:-1], np.abs(prices[1:] - prices[1:-1]), np.abs(prices[:-1] - prices[:-2]))
        atr = np.mean(tr[-n:]) if len(tr) >= n else np.mean(tr)
        return atr
    
    atr = compute_atr(closing_prices)
    features.append(atr if not np.isnan(atr) else 0)

    # Feature 2: Momentum (Rate of Change)
    momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if closing_prices[-6] != 0 else 0
    features.append(momentum)

    # Feature 3: Standard Deviation of Returns for Volatility
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    volatility = np.std(returns) if len(returns) > 0 else 0
    features.append(volatility)

    # Feature 4: Recent Drawdown
    recent_drawdown = np.max(closing_prices) - closing_prices[-1]  # Current price vs max price
    features.append(recent_drawdown / np.max(closing_prices) if np.max(closing_prices) != 0 else 0)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    features = enhanced_s[123:]
    reward = 0.0

    # Calculate dynamic thresholds based on feature statistics
    avg_volatility = np.mean(features[2])  # Assuming features[2] represents volatility
    high_vol_threshold = avg_volatility + (2 * np.std(features[2]))

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50  # STRONG NEGATIVE for BUY-aligned features
        reward += 10 if features[0] < high_vol_threshold else 0  # MILD POSITIVE for SELL if volatility is low
    
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += 10 * features[1]  # Reward based on momentum
        elif trend_direction < -0.3:  # Downtrend
            reward += 10 * -features[1]  # Reward based on momentum in the opposite direction

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        # Assuming features[1] indicates mean-reversion potential
        if features[1] < 0:  # Oversold condition
            reward += 10  # Reward for potential buying opportunity
        elif features[1] > 0:  # Overbought condition
            reward += -10  # Penalize for potential selling opportunity

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > high_vol_threshold and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within bounds