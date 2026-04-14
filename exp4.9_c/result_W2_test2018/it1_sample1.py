import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extract closing prices
    volumes = s[4::6]          # Extract trading volumes
    
    # Feature 1: Bollinger Bands
    window = 20
    if len(closing_prices) > window:
        rolling_mean = np.mean(closing_prices[-window:])
        rolling_std = np.std(closing_prices[-window:])
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)
        price_below_lower_band = (closing_prices[-1] < lower_band).astype(float)
        price_above_upper_band = (closing_prices[-1] > upper_band).astype(float)
    else:
        price_below_lower_band = 0.0
        price_above_upper_band = 0.0

    # Feature 2: Exponential Moving Average (EMA)
    ema_period = 12
    if len(closing_prices) >= ema_period:
        ema = np.mean(closing_prices[-ema_period:])  # Simplified EMA for demonstration
    else:
        ema = closing_prices[-1] if len(closing_prices) > 0 else 0

    # Feature 3: Z-score of recent returns
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    if len(returns) > 20:  # Use past 20 for Z-score
        mean_return = np.mean(returns[-20:])
        std_return = np.std(returns[-20:])
        z_score = (returns[-1] - mean_return) / (std_return + 1e-10)  # Avoid division by zero
    else:
        z_score = 0.0

    features = [price_below_lower_band, price_above_upper_band, ema, z_score]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]
    
    reward = 0.0

    # Priority 1: Risk management
    if risk_level > 0.7:
        reward -= 50  # Strong negative for high-risk BUY signals
        reward += 10 * (1 - enhanced_s[123])  # Mild positive for SELL signals (if features indicate sell)
        return np.clip(reward, -100, 100)

    if risk_level > 0.4:
        reward -= 20  # Moderate negative for BUY signals

    # Priority 2: Trend Following
    if abs(trend_direction) > 0.3 and risk_level < 0.4:
        if trend_direction > 0:  # Uptrend
            reward += 20 if enhanced_s[123][0] > 0 else -10  # Positive for momentum
        else:  # Downtrend
            reward += 20 if enhanced_s[123][0] < 0 else -10  # Positive for momentum

    # Priority 3: Sideways / Mean Reversion
    elif abs(trend_direction) < 0.3 and risk_level < 0.3:
        reward += 10 if enhanced_s[123][0] < 0 else -5  # Reward mean-reversion features

    # Priority 4: High Volatility
    if volatility_level > 0.6 and risk_level < 0.4:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]