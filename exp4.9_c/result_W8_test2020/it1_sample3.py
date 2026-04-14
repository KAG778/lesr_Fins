import numpy as np

def revise_state(s):
    features = []
    closing_prices = s[0::6]  # Extract closing prices
    
    # 1. Exponential Moving Average (EMA) for trend detection
    ema_short = np.mean(closing_prices[-12:]) if len(closing_prices) >= 12 else 0.0
    ema_long = np.mean(closing_prices[-26:]) if len(closing_prices) >= 26 else 0.0
    ema = ema_short - ema_long  # MACD-like feature for trend
    
    features.append(ema)
    
    # 2. Average True Range (ATR) for volatility assessment
    high_prices = s[1::6]
    low_prices = s[2::6]
    true_ranges = np.maximum(high_prices[1:] - low_prices[1:], 
                             np.maximum(np.abs(high_prices[1:] - closing_prices[:-1]), 
                                        np.abs(low_prices[1:] - closing_prices[:-1])))
    atr = np.mean(true_ranges[-14:]) if len(true_ranges) >= 14 else 0.0  # 14-day ATR
    features.append(atr)

    # 3. Stochastic Oscillator for overbought/oversold conditions
    lowest_low = np.min(low_prices[-14:]) if len(low_prices) >= 14 else 0.0
    highest_high = np.max(high_prices[-14:]) if len(high_prices) >= 14 else 0.0
    current_close = closing_prices[-1] if len(closing_prices) > 0 else 0.0
    stochastic = 100 * (current_close - lowest_low) / (highest_high - lowest_low) if highest_high != lowest_low else 0.0
    features.append(stochastic)

    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0
    relative_thresholds = {
        'risk_high': 0.7,
        'risk_moderate': 0.4,
        'trend_threshold': 0.3,
        'volatility_threshold': 0.6
    }

    # Adjusting thresholds based on historical std dev
    risk_high_threshold = np.mean([risk_level]) + 1.5 * np.std([risk_level])
    risk_moderate_threshold = np.mean([risk_level]) + 0.5 * np.std([risk_level])
    
    # Priority 1: Risk Management
    if risk_level > risk_high_threshold:
        reward -= np.random.uniform(30, 50)  # Strong negative reward for BUY actions
    elif risk_level > risk_moderate_threshold:
        reward -= np.random.uniform(10, 20)  # Moderate negative reward for BUY actions

    # Priority 2: Trend Following (when risk is low)
    elif abs(trend_direction) > relative_thresholds['trend_threshold'] and risk_level < risk_moderate_threshold:
        if trend_direction > 0:  # Uptrend
            reward += 10  # Positive reward for BUY signals
        elif trend_direction < 0:  # Downtrend
            reward += 10  # Positive reward for SELL signals

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < relative_thresholds['trend_threshold'] and risk_level < relative_thresholds['risk_moderate']:
        reward += 5  # Reward for mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > relative_thresholds['volatility_threshold'] and risk_level < risk_moderate_threshold:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return float(np.clip(reward, -100, 100))  # Ensure the reward is within [-100, 100]