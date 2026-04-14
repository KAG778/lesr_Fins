import numpy as np

def revise_state(s):
    # s: 120d raw state (20 days of OHLCV data)
    n_days = 20
    closing_prices = s[0::6][:n_days]  # Extract closing prices
    volumes = s[4::6][:n_days]          # Extract trading volumes
    
    # Feature 1: Exponential Moving Average (EMA) for price (span: 14 days)
    def compute_ema(prices, span):
        weights = np.exp(np.linspace(-1., 0., span))
        weights /= weights.sum()
        ema = np.convolve(prices, weights, mode='full')[:len(prices)]
        ema[:span] = np.nan  # Set the initial values to NaN for proper alignment
        return ema[-1]  # Return the latest EMA value

    ema = compute_ema(closing_prices, span=14)

    # Feature 2: Rate of Change (ROC) over the last 14 days
    roc = (closing_prices[-1] - closing_prices[-15]) / closing_prices[-15] if n_days > 14 else 0

    # Feature 3: Average True Range (ATR) for volatility (14 days)
    high_prices = s[2::6][:n_days]
    low_prices = s[3::6][:n_days]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                              np.maximum(np.abs(high_prices[1:] - closing_prices[:-1][1:]), 
                                         np.abs(low_prices[1:] - closing_prices[:-1][1:])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0  # ATR over the last 14 days

    # Feature 4: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0

    # Compile features into a single array
    features = [ema, roc, atr, vwap]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate relative thresholds for risk and volatility based on historical std
    risk_threshold = 0.5  # Assuming a base threshold for mid-risk as an example
    volatility_threshold = 0.5  # Assuming a base threshold for mid-volatility as an example

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold:
        # Strong negative reward for BUY-aligned features
        reward -= np.random.uniform(30, 50)  # STRONG NEGATIVE reward for BUY
        # Mild positive reward for SELL-aligned features
        reward += np.random.uniform(5, 10)  # MILD POSITIVE for SELL
    
    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < risk_threshold:
        if trend_direction > 0:
            reward += np.random.uniform(10, 20)  # Positive reward for upward features
        else:
            reward += np.random.uniform(10, 20)  # Positive reward for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += np.random.uniform(5, 15)  # Reward mean-reversion features
        reward -= np.random.uniform(5, 10)  # Penalize breakout-chasing features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > volatility_threshold and risk_level < risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds