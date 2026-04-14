import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0::6]  # Extract closing prices every 6th element
    volumes = s[4::6]          # Extract volumes every 6th element

    days = len(closing_prices)

    # Feature 1: Average True Range (ATR)
    def compute_atr(prices, period=14):
        high = np.array(prices) + np.random.rand(len(prices)) * 0.01  # Simulate high prices for the sake of ATR
        low = np.array(prices) - np.random.rand(len(prices)) * 0.01  # Simulate low prices
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(prices, 1)), np.abs(low - np.roll(prices, 1))))
        atr = np.mean(tr[-period:]) if len(tr) >= period else 0
        return atr

    atr = compute_atr(closing_prices)
    
    # Feature 2: Bollinger Bands (20-day SMA and Std Dev)
    def compute_bollinger_bands(prices, window=20):
        sma = np.mean(prices[-window:]) if len(prices) >= window else 0
        std_dev = np.std(prices[-window:]) if len(prices) >= window else 0
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        return upper_band, lower_band, sma

    upper_band, lower_band, sma = compute_bollinger_bands(closing_prices)

    # Feature 3: Change in Volume
    volume_change = (volumes[-1] - np.mean(volumes[-14:])) / np.mean(volumes[-14:]) if np.mean(volumes[-14:]) != 0 else 0

    # Feature 4: EMA Crossover (12 and 26 days)
    def compute_ema(prices, span):
        return prices.ewm(span=span, adjust=False).mean().values[-1]

    ema12 = compute_ema(closing_prices, 12)
    ema26 = compute_ema(closing_prices, 26)
    ema_crossover = ema12 - ema26

    features = [
        atr,
        upper_band,
        lower_band,
        volume_change,
        ema_crossover
    ]
    
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    # Calculate historical thresholds based on past data
    risk_threshold_high = 0.7  # Replace with dynamic threshold if historical data is available
    risk_threshold_medium = 0.4  # Replace with dynamic threshold if historical data is available
    
    reward = 0.0
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > risk_threshold_high:
        reward -= 50  # Strong negative reward for BUY
    elif risk_level > risk_threshold_medium:
        reward -= 15  # Mild negative for BUY

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_threshold_medium:
        reward += 20 * np.sign(trend_direction)  # Reward momentum alignment

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_threshold_medium:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)