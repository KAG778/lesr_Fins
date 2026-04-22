import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]  # Extract trading volumes
    
    # Feature 1: Price Movement (percentage change)
    latest_price = closing_prices[-1]
    past_price = closing_prices[-20] if len(closing_prices) >= 20 else 0
    price_change = (latest_price - past_price) / (past_price if past_price != 0 else 1)  # Avoid division by zero

    # Feature 2: Volume Change (percentage change)
    latest_volume = volumes[-1]
    past_volume = volumes[-20] if len(volumes) >= 20 else 0
    volume_change = (latest_volume - past_volume) / (past_volume if past_volume != 0 else 1)  # Avoid division by zero

    # Feature 3: MACD (12-day EMA - 26-day EMA)
    def ema(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    ema12 = ema(closing_prices, 12)[-1] if len(closing_prices) >= 12 else 0
    ema26 = ema(closing_prices, 26)[-1] if len(closing_prices) >= 26 else 0
    macd = ema12 - ema26

    features = [price_change, volume_change, macd]
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
        reward += -40  # Strong negative reward for BUY-aligned features
    if risk_level > 0.4:
        reward += -20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward features
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 5  # Reward mean-reversion features
        reward -= 5  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds