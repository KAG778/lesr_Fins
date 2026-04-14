import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices

    # 1. Rate of Change (ROC)
    roc = (closing_prices[-1] - closing_prices[-2]) / closing_prices[-2] if closing_prices[-2] != 0 else 0.0
    features.append(roc)

    # 2. Relative Strength Index (RSI)
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Mean gain
    loss = np.abs(np.where(delta < 0, delta, 0)).mean()  # Mean loss
    rs = gain / loss if loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    features.append(rsi)

    # 3. Moving Average Convergence Divergence (MACD)
    short_term_ema = np.mean(closing_prices[-12:])  # Short-term EMA (last 12 days)
    long_term_ema = np.mean(closing_prices[-26:])  # Long-term EMA (last 26 days)
    macd = short_term_ema - long_term_ema
    features.append(macd)

    # 4. Historical Volatility (standard deviation of closing prices over last 20 days)
    historical_volatility = np.std(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    features.append(historical_volatility)

    # 5. Mean Reversion Indicator (distance from moving average)
    moving_average = np.mean(closing_prices[-20:]) if len(closing_prices) >= 20 else 0.0
    mean_reversion_indicator = closing_prices[-1] - moving_average
    features.append(mean_reversion_indicator)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds for risk
    historical_volatility = np.std(enhanced_s[123:, 3])  # Assuming feature 3 is historical volatility
    low_risk_threshold = 0.4 * historical_volatility
    high_risk_threshold = 0.7 * historical_volatility

    # Priority 1: Risk Management
    if risk_level > high_risk_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY actions
    elif risk_level > low_risk_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY actions

    # Priority 2: Trend Following (when risk is low)
    elif abs(trend_direction) > 0.3 and risk_level <= low_risk_threshold:
        if trend_direction > 0.3:
            reward += 10  # Positive reward for upward trend
        elif trend_direction < -0.3:
            reward += 10  # Positive reward for downward trend

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level <= low_risk_threshold:
        # Reward for mean-reversion features
        mean_reversion_value = enhanced_s[123, 4]  # Assuming feature 4 is mean-reversion indicator
        oversold_threshold = -0.02 * historical_volatility  # Relative threshold for oversold
        overbought_threshold = 0.02 * historical_volatility  # Relative threshold for overbought
        if mean_reversion_value < oversold_threshold:
            reward += 15.0  # Reward for buying an oversold signal
        elif mean_reversion_value > overbought_threshold:
            reward += 15.0  # Reward for selling an overbought signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level <= low_risk_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within [-100, 100]