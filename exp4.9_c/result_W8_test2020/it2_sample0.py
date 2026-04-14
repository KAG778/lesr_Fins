import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices

    # 1. Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0.0
    ema_long = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0.0
    ema = ema_short - ema_long
    features.append(ema)

    # 2. Average True Range (ATR) for volatility assessment
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0
    features.append(atr)

    # 3. Stochastic Oscillator for overbought/oversold conditions
    lowest_low = np.min(low_prices[-14:]) if len(low_prices) >= 14 else 0.0
    highest_high = np.max(high_prices[-14:]) if len(high_prices) >= 14 else 0.0
    current_close = closing_prices[-1] if len(closing_prices) > 0 else 0.0
    stochastic = 100 * (current_close - lowest_low) / (highest_high - lowest_low) if highest_high != lowest_low else 0.0
    features.append(stochastic)

    # 4. Z-score of returns for mean-reversion signals
    daily_returns = np.diff(closing_prices) / closing_prices[:-1]
    mean_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0.0
    std_return = np.std(daily_returns) if len(daily_returns) > 0 else 1.0  # Avoid division by zero
    z_score = (daily_returns[-1] - mean_return) / std_return if std_return != 0 else 0
    features.append(z_score)

    # 5. Momentum calculation (current price - price N days ago)
    N = 5
    momentum = closing_prices[-1] - closing_prices[-N] if len(closing_prices) >= N else 0.0
    features.append(momentum)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate relative thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Compute std from features
    risk_high_threshold = 0.7 * historical_std
    risk_moderate_threshold = 0.4 * historical_std
    trend_threshold = 0.3 * historical_std

    # Priority 1: Risk Management
    if risk_level > risk_high_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative for BUY actions
    elif risk_level > risk_moderate_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative for BUY actions

    # Priority 2: Trend Following (when risk is low)
    elif abs(trend_direction) > trend_threshold and risk_level <= risk_moderate_threshold:
        if trend_direction > 0:  # Uptrend
            reward += 10  # Positive reward for upward momentum
        elif trend_direction < 0:  # Downtrend
            reward += 10  # Positive reward for downward momentum

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < trend_threshold and risk_level < 0.3:
        if enhanced_s[123] < 30:  # Oversold condition
            reward += 15  # Reward for buying an oversold signal
        elif enhanced_s[123] > 70:  # Overbought condition
            reward += 15  # Reward for selling an overbought signal

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level <= risk_moderate_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50% during high volatility

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within [-100, 100]