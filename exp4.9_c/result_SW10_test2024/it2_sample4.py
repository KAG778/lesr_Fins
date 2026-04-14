import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    features = []

    # 1. Crisis Indicator: Percentage drop from the peak in the last 20 days
    peak_price = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    current_price = closing_prices[-1]
    crisis_indicator = ((peak_price - current_price) / peak_price) * 100  # Percentage drop
    features.append(crisis_indicator)

    # 2. 20-day Average Volume
    avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    features.append(avg_volume)

    # 3. 14-day Relative Strength Index (RSI)
    def compute_rsi(prices, period=14):
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        average_gain = np.mean(gain[-period:]) if len(gain) >= period else 0
        average_loss = np.mean(loss[-period:]) if len(loss) >= period else 0
        if average_loss == 0:
            return 100  # Avoid division by zero
        rs = average_gain / average_loss
        return 100 - (100 / (1 + rs))

    rsi = compute_rsi(closing_prices)
    features.append(rsi)

    # 4. Moving Average Convergence Divergence (MACD)
    def compute_macd(prices, short_window=12, long_window=26):
        short_ema = np.mean(prices[-short_window:]) if len(prices) >= short_window else prices[-1]
        long_ema = np.mean(prices[-long_window:]) if len(prices) >= long_window else prices[-1]
        return short_ema - long_ema

    macd = compute_macd(closing_prices)
    features.append(macd)

    # 5. Average True Range (ATR) for volatility
    def compute_atr(prices, period=14):
        tr = np.zeros(len(prices)-1)
        for i in range(1, len(prices)):
            tr[i-1] = max(prices[i] - prices[i-1], 
                           abs(prices[i] - prices[i-1]), 
                           abs(prices[i-1] - prices[i]))
        return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

    atr = compute_atr(closing_prices)
    features.append(atr)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical standard deviation for relative thresholds
    feature_volatility = np.std(enhanced_s[123:])  # Use feature array for volatility
    high_vol_threshold = feature_volatility * 1.5
    low_vol_threshold = feature_volatility * 0.5
    
    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10 if trend_direction < 0 else -10  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 25 if trend_direction > 0 else 15  # Strong reward for upward momentum, moderate for downward

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3:
        reward += 20  # Reward potential mean-reversion features
        reward -= 10  # Penalize for chasing breakouts

    # Priority 4: HIGH VOLATILITY
    if volatility_level > high_vol_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%
    elif volatility_level < low_vol_threshold:
        reward *= 2.0  # Potentially increase reward if volatility is low

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds