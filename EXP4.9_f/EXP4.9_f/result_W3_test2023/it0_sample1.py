import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6][:20]  # Extract closing prices
    volumes = s[4::6][:20]          # Extract trading volumes

    # Feature 1: 5-day Simple Moving Average (SMA)
    sma_5 = np.mean(closing_prices[-5:]) if len(closing_prices) >= 5 else np.nan

    # Feature 2: 10-day Simple Moving Average (SMA)
    sma_10 = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else np.nan

    # Feature 3: Price Momentum (current price - price 5 days ago)
    momentum = closing_prices[-1] - closing_prices[-6] if len(closing_prices) >= 6 else np.nan

    # Feature 4: Volume Change (current volume - average volume over the last 5 days)
    avg_volume_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else np.nan
    volume_change = volumes[-1] - avg_volume_5 if avg_volume_5 != 0 else np.nan

    # Handle NaN values - replace with 0 or another suitable value
    features = [
        sma_5 if not np.isnan(sma_5) else 0,
        sma_10 if not np.isnan(sma_10) else 0,
        momentum if not np.isnan(momentum) else 0,
        volume_change if not np.isnan(volume_change) else 0
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    # enhanced_s[0:120] = raw state
    # enhanced_s[120:123] = regime_vector
    # enhanced_s[123:] = your features
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE for BUY-aligned signals
    elif risk_level > 0.4:
        reward -= 10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:
            reward += 10  # Positive reward for upward features
        else:
            reward += 10  # Positive reward for downward features (correct bearish bet)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features
        reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]