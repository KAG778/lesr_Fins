import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract volumes

    features = []

    # Feature 1: Price Momentum (current close - close 20 days ago)
    if len(closing_prices) > 20:
        price_momentum = closing_prices[0] - closing_prices[20]
    else:
        price_momentum = 0  # Handle edge case
    features.append(price_momentum)

    # Feature 2: Average True Range (ATR)
    if len(closing_prices) > 1:
        high_prices = s[2::6]  # Extract high prices
        low_prices = s[3::6]   # Extract low prices
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - closing_prices[:-1]), 
                                   abs(low_prices[1:] - closing_prices[:-1])))
        atr = np.mean(tr) if len(tr) > 0 else 0
    else:
        atr = 0
    features.append(atr)

    # Feature 3: Bollinger Bands (using the last 20 days)
    if len(closing_prices) >= 20:
        rolling_mean = np.mean(closing_prices[:20])
        rolling_std = np.std(closing_prices[:20])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        features.append(upper_band)
        features.append(lower_band)
    else:
        features.append(0)  # Default if not enough data
        features.append(0)

    # Feature 4: Volume-weighted Average Price (VWAP)
    if len(volumes) > 0:
        vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0
    else:
        vwap = 0
    features.append(vwap)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]
    reward = 0.0

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        if features[0] > 0:  # Assume positive feature indicates BUY
            reward = np.random.uniform(-50, -30)  # Strong negative for BUY
        else:
            reward = np.random.uniform(5, 10)  # Mild positive for SELL
    elif risk_level > 0.4:
        if features[0] > 0:  # Assume positive feature indicates BUY
            reward = -10  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:  # Uptrend
            reward += min(20, features[0])  # Positive reward for upward features
        elif trend_direction < -0.3:  # Downtrend
            reward += min(20, -features[0])  # Positive reward for downward features

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[2] < features[3]:  # Assuming lower band < upper band indicates mean reversion
            reward += 15  # Reward for mean-reversion features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds