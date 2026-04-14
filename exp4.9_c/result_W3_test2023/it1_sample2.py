import numpy as np

def revise_state(s):
    # Extracting closing prices and volumes
    closing_prices = s[0::6]  # Closing prices
    volumes = s[4::6]         # Trading volumes

    features = []

    # Feature 1: Average True Range (ATR) for volatility measure
    def calculate_atr(prices, high_prices, low_prices, period=14):
        tr = np.maximum(high_prices[1:] - low_prices[1:], 
                        np.maximum(abs(high_prices[1:] - prices[:-1]), 
                                   abs(low_prices[1:] - prices[:-1])))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.nan
        return atr

    # Calculate ATR
    high_prices = s[2::6]
    low_prices = s[3::6]
    atr_value = calculate_atr(closing_prices, high_prices, low_prices)
    features.append(atr_value if atr_value is not np.nan else 0)

    # Feature 2: 14-day RSI
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0).mean() if len(deltas) >= period else 0
        losses = -np.where(deltas < 0, deltas, 0).mean() if len(deltas) >= period else 0
        rs = gains / losses if losses != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi_value = calculate_rsi(closing_prices)
    features.append(rsi_value)

    # Feature 3: Rate of Change (ROC) of closing prices
    roc = (closing_prices[-1] - closing_prices[-10]) / closing_prices[-10] if len(closing_prices) > 10 else 0
    features.append(roc)

    # Feature 4: Volume Weighted Average Price (VWAP)
    vwaps = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0
    features.append(vwaps)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    # Extract regime information
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Initialize reward
    reward = 0.0

    # Calculate historical standard deviation for relative thresholds
    historical_prices = enhanced_s[0:120:6]
    historical_std = np.std(historical_prices)
    
    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -50  # Strong negative for BUY-aligned features
    elif risk_level > 0.4:
        reward += -10  # Moderate negative for BUY signals

    # Priority 2 — TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0.3:
            reward += 20  # Positive reward for buying in an uptrend
        elif trend_direction < -0.3:
            reward += 20  # Positive reward for selling in a downtrend

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]