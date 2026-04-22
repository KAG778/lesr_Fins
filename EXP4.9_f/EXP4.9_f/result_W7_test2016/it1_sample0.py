import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    features = []

    # Feature 1: Relative Strength Index (RSI) over the last 14 days
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(closing_prices)
    features.append(rsi)

    # Feature 2: Bollinger Bands
    def calculate_bollinger_bands(prices, window=20, num_sd=2):
        if len(prices) < window:
            return 0, 0  # Not enough data
        
        sma = np.mean(prices[-window:])
        rstd = np.std(prices[-window:])
        upper_band = sma + (num_sd * rstd)
        lower_band = sma - (num_sd * rstd)
        return upper_band, lower_band
    
    upper_band, lower_band = calculate_bollinger_bands(closing_prices)
    features.append(upper_band)
    features.append(lower_band)

    # Feature 3: Market Breadth (e.g., number of stocks above their 50-day MA)
    # Here we simulate this feature as a placeholder; in practice, you would pull this from a market data source.
    breadth_indicator = np.random.uniform(-1, 1)  # Placeholder for actual breadth computation
    features.append(breadth_indicator)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate relative thresholds based on historical data
    historical_std = 0.2  # Placeholder for historical std deviation (this should be calculated based on past data)
    threshold_risk_high = 0.7 * historical_std
    threshold_risk_medium = 0.4 * historical_std
    threshold_trend = 0.3 * historical_std

    # Initialize reward
    reward = 0.0

    # Priority 1 — RISK MANAGEMENT
    if risk_level > threshold_risk_high:
        reward -= 50  # Strong negative for BUY
        reward += 10   # Mild positive for SELL
    elif risk_level > threshold_risk_medium:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > threshold_trend and risk_level < threshold_risk_medium:
        if trend_direction > threshold_trend:
            reward += 30  # Strong positive for upward features
        elif trend_direction < -threshold_trend:
            reward += 30  # Strong positive for downward features

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < threshold_trend and risk_level < 0.3:
        reward += 20  # Reward for mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < threshold_risk_medium:
        reward *= 0.5  # Reduce reward magnitude

    # Ensure reward is within [-100, 100]
    return np.clip(reward, -100, 100)