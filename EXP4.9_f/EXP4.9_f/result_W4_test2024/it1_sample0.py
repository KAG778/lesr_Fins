import numpy as np

def revise_state(s):
    # s: 120d raw state
    closing_prices = s[0:120:6]  # Extract closing prices (s[i*6 + 0] for i=0..19)
    volumes = s[4:120:6]         # Extract volumes for 20 days
    days = len(closing_prices)

    # Feature 1: Exponential Moving Average (EMA) of Closing Prices
    def ema(prices, window):
        if len(prices) < window:
            return 0
        return np.mean(prices[-window:])  # Simple moving average as a proxy for EMA

    short_ema = ema(closing_prices, 5)
    long_ema = ema(closing_prices, 20)
    ema_crossover = short_ema - long_ema  # Positive if short EMA is above long EMA

    # Feature 2: Average True Range (ATR) for Volatility
    def atr(prices, high, low, window):
        tr = np.maximum(high - low, np.abs(high - np.roll(prices, 1)), np.abs(low - np.roll(prices, 1)))
        return np.mean(tr[-window:]) if len(tr) >= window else 0

    atr_value = atr(closing_prices, s[2:120:6], s[3:120:6], 14)  # 14-day ATR

    # Feature 3: Volume Weighted Average Price (VWAP)
    vwap = np.sum(closing_prices * volumes) / np.sum(volumes) if np.sum(volumes) != 0 else 0

    # Feature 4: Price Relative to VWAP
    price_relative_to_vwap = (closing_prices[-1] - vwap) / vwap if vwap != 0 else 0

    features = [ema_crossover, atr_value, price_relative_to_vwap]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical thresholds based on risk level
    risk_thresholds = [0.3, 0.7]  # Example thresholds for low and high risk
    avg_risk = np.mean([risk_level for _ in range(100)])  # Replace with actual historical risk calculation

    reward = 0
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > risk_thresholds[1]:
        reward -= 40  # Strong negative for BUY-aligned features
        # Mild positive for SELL-aligned features (for risk management)
        reward += 10 if trend_direction < 0 else 0
    elif risk_level > risk_thresholds[0]:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    if abs(trend_direction) > 0.3 and risk_level < risk_thresholds[0]:
        reward += 20 * np.sign(trend_direction)  # Positive reward for trend-following

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10  # Reward for mean-reversion features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > 0.6 and risk_level < risk_thresholds[0]:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure reward is within [-100, 100]