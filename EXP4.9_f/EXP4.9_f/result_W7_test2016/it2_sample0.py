import numpy as np

def revise_state(s):
    features = []
    
    # Extract closing prices and volumes
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes

    # Feature 1: Average True Range (ATR) over the last 14 days
    def calculate_atr(prices, window=14):
        high = prices[2::6]
        low = prices[3::6]
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prices[:-1:6]), np.abs(low[1:] - prices[:-1:6])))
        atr = np.mean(tr[-window:]) if len(tr) >= window else 0
        return atr
    
    atr = calculate_atr(s)
    features.append(atr)

    # Feature 2: Williams %R (a momentum indicator)
    def calculate_williams_r(prices, window=14):
        highest_high = np.max(prices[-window:]) if len(prices) >= window else prices[-1]
        lowest_low = np.min(prices[-window:]) if len(prices) >= window else prices[-1]
        williams_r = (highest_high - prices[-1]) / (highest_high - lowest_low + 1e-10) * -100
        return williams_r
    
    williams_r = calculate_williams_r(closing_prices)
    features.append(williams_r)

    # Feature 3: Price Momentum (latest closing price compared to 5 days ago)
    price_momentum = closing_prices[-1] - closing_prices[-5] if len(closing_prices) >= 5 else 0
    features.append(price_momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[0:120])  # Calculate std deviation from raw state
    risk_threshold = 0.7 * historical_std
    trend_threshold = 0.3 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        reward -= 50  # Strong negative for BUY
        reward += 10 if enhanced_s[123] < 0 else 0  # Mild positive for SELL
    elif risk_level > 0.4 * historical_std:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level < 0.4 * historical_std:
        if trend_direction > trend_threshold:
            reward += 30  # Strong positive for upward features
        elif trend_direction < -trend_threshold:
            reward += 30  # Strong positive for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3 * historical_std:
        reward += 20  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4 * historical_std:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)