import numpy as np

def revise_state(s):
    closing_prices = s[0:120:6]  # Extract closing prices
    volumes = s[4:120:6]          # Extract trading volumes
    features = []

    # 1. Price Change Rate (percentage change over the last 5 days)
    if len(closing_prices) >= 6:
        price_change_rate = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6]
    else:
        price_change_rate = 0  # Not enough data
    features.append(price_change_rate)

    # 2. Relative Strength Index (RSI) - 14-day
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

    # 3. 20-day historical volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Calculate daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0
    features.append(historical_volatility)

    # 4. Crisis Indicator (percentage drop from peak in the last 20 days)
    peak_price = np.max(closing_prices[-20:]) if len(closing_prices) >= 20 else closing_prices[-1]
    current_price = closing_prices[-1]
    crisis_indicator = ((peak_price - current_price) / peak_price) * 100  # Percentage drop
    features.append(crisis_indicator)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate historical thresholds for risk management
    historical_volatility = np.std(enhanced_s[123:])  # Use features for volatility
    high_vol_threshold = historical_volatility * 1.5  # Define high volatility threshold
    low_vol_threshold = historical_volatility * 0.5   # Define low volatility threshold

    # Priority 1: RISK MANAGEMENT
    if risk_level > 0.7:
        reward += -40  # Strong negative for BUY-aligned features
        reward += 10 if trend_direction < 0 else -10  # Mild positive for SELL-aligned features
    elif risk_level > 0.4:
        reward += -20  # Moderate negative for BUY signals

    # Priority 2: TREND FOLLOWING
    elif abs(trend_direction) > 0.3 and risk_level < 0.4:
        reward += 20 * np.sign(trend_direction)  # Align reward with trend direction

    # Priority 3: SIDEWAYS / MEAN REVERSION
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 15  # Reward for potential mean-reversion features
        reward -= 5   # Penalize for chasing breakouts

    # Priority 4: HIGH VOLATILITY
    if volatility_level > high_vol_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%
    elif volatility_level < low_vol_threshold:
        reward *= 1.5  # Increase reward if volatility is low

    return np.clip(reward, -100, 100)  # Ensure reward is within bounds