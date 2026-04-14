import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices (20 days)
    volumes = s[4:120:6]          # Extract trading volumes
    features = []

    # 1. Daily Price Change Percentage
    if len(closing_prices) >= 2:
        price_change_pct = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2]
    else:
        price_change_pct = 0
    features.append(price_change_pct)

    # 2. Average True Range (ATR) for volatility
    def compute_atr(prices, period=14):
        tr = np.zeros(len(prices) - 1)
        for i in range(1, len(prices)):
            tr[i - 1] = max(prices[i] - prices[i - 1], 
                            abs(prices[i] - prices[i - 1]), 
                            abs(prices[i - 1] - prices[i]))
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    atr = compute_atr(closing_prices)
    features.append(atr)

    # 3. Crisis Indicator (percentage drop from peak in the last 20 days)
    peak_price = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    current_price = closing_prices[-1]
    crisis_indicator = ((peak_price - current_price) / peak_price) * 100 if peak_price > 0 else 0
    features.append(crisis_indicator)

    # 4. 20-day moving average of closing prices
    if len(closing_prices) >= 20:
        moving_average = np.mean(closing_prices[-20:])
    else:
        moving_average = closing_prices[-1]  # Fallback to last price
    features.append(moving_average)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    # Calculate historical volatility for relative thresholds
    historical_std = np.std(enhanced_s[123:])  # Use features for volatility
    high_vol_threshold = historical_std * 1.5  # Example threshold for high volatility

    reward = 0.0

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10 if trend_direction < 0 else -10  # MILD POSITIVE for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 20 * np.sign(trend_direction)  # Positive reward aligned with trend direction

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward potential mean-reversion features

    # Priority 4: HIGH VOLATILITY
    if volatility_level > high_vol_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds