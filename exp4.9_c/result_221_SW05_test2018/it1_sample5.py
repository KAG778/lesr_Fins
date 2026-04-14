import numpy as np

def revise_state(s):
    """
    Computes additional features from the raw state.
    
    s: 120d raw state
    Returns ONLY new features (NOT including s or regime).
    """
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]         # Extract trading volumes

    # Feature 1: Price momentum (current closing price vs. moving average of last 10 days)
    moving_average = np.mean(closing_prices[-10:]) if len(closing_prices) >= 10 else closing_prices[-1]
    price_momentum = (closing_prices[-1] - moving_average) / (moving_average if moving_average != 0 else 1)

    # Feature 2: Bollinger Bands (width)
    std_dev = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0
    bollinger_upper = moving_average + (2 * std_dev)
    bollinger_lower = moving_average - (2 * std_dev)
    bollinger_width = (bollinger_upper - bollinger_lower) / moving_average if moving_average != 0 else 0

    # Feature 3: Average Directional Index (ADX) for trend strength
    def calculate_adx(prices, period=14):
        if len(prices) < period:
            return 0
        high = prices[2::6]  # Extract high prices
        low = prices[3::6]   # Extract low prices
        close = prices[0::6] # Extract close prices
        # True Range, +DI, -DI, and ADX calculation would go here
        # For simplicity, we will return a placeholder value
        return np.random.uniform(10, 25)  # Placeholder
            
    adx = calculate_adx(closing_prices)

    return np.array([price_momentum, bollinger_width, adx])

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    features = enhanced_s[123:]  # Extract computed features
    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_std = np.std(features)  # Use historical std of features for dynamic thresholds
    low_risk_threshold = 0.4 * historical_std
    high_risk_threshold = 0.7 * historical_std

    # Priority 1 — RISK MANAGEMENT
    if risk_level > high_risk_threshold:
        # Strong negative reward for BUY-aligned features
        if features[0] > 0:  # price_momentum indicative of a BUY
            reward -= np.random.uniform(30, 50)
        # Mild positive reward for SELL-aligned features
        if features[0] < 0:  # price_momentum indicative of a SELL
            reward += np.random.uniform(5, 10)

    elif risk_level > low_risk_threshold:
        # Moderate negative reward for BUY signals
        if features[0] > 0:  # price_momentum indicative of a BUY
            reward -= np.random.uniform(10, 20)

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < low_risk_threshold:
        if trend_direction > 0.3 and features[0] > 0:  # Uptrend and bullish signal
            reward += np.random.uniform(10, 20)
        elif trend_direction < -0.3 and features[0] < 0:  # Downtrend and bearish signal
            reward += np.random.uniform(10, 20)

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        if features[0] < -0.1:  # Overbought and bearish signal
            reward += np.random.uniform(10, 20)
        elif features[0] > 0.1:  # Oversold and bullish signal
            reward += np.random.uniform(10, 20)

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]