import numpy as np

def revise_state(s):
    closing_prices = s[0::6]  # Extracting closing prices
    high_prices = s[2::6]     # Extracting high prices
    low_prices = s[3::6]      # Extracting low prices
    volumes = s[4::6]         # Extracting trading volumes

    # Feature 1: Price Momentum (current close - close 5 days ago)
    price_momentum = (closing_prices[-1] - closing_prices[-6]) / closing_prices[-6] if len(closing_prices) > 5 else 0

    # Feature 2: Average True Range (ATR) for volatility measurement
    tr = np.maximum(high_prices[1:] - low_prices[1:], 
                    high_prices[1:] - closing_prices[:-1], 
                    closing_prices[:-1] - low_prices[1:])
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0  # 14-day ATR

    # Feature 3: Relative Strength Index (RSI) calculation
    delta = np.diff(closing_prices)
    gain = np.where(delta > 0, delta, 0).mean()  # Average gain
    loss = np.where(delta < 0, -delta, 0).mean()  # Average loss
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))  # RSI formula

    # Feature 4: 20-day Historical Volatility (standard deviation of returns)
    returns = np.diff(closing_prices) / closing_prices[:-1]  # Daily returns
    historical_volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0  # Volatility over the last 20 days

    # Feature 5: Percentage of Volume Change (current volume / mean volume over last 5 days)
    avg_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1  # To avoid division by zero
    volume_change = (volumes[-1] - avg_volume) / avg_volume if avg_volume != 0 else 0

    features = [price_momentum, atr, rsi, historical_volatility, volume_change]
    return np.array(features)

def intrinsic_reward(enhanced_s):
    regime = enhanced_s[120:123]
    trend_direction = regime[0]
    volatility_level = regime[1]
    risk_level = regime[2]

    reward = 0.0

    # Calculate dynamic thresholds based on historical data
    historical_std = np.std(enhanced_s[123:])  # Using features for dynamic thresholding
    risk_threshold = 0.5 * historical_std  # Dynamic threshold for risk
    trend_threshold = 0.3 * historical_std  # Dynamic threshold for trend detection

    # Priority 1 — RISK MANAGEMENT
    if risk_level > 0.7:
        reward -= 50  # STRONG NEGATIVE for high-risk BUY-aligned features
        reward += 10   # Mild positive reward for SELL-aligned features
        return np.clip(reward, -100, 100)  # Early return if in high-risk environment
    elif risk_level > risk_threshold:
        reward -= 20  # Moderate negative reward for BUY signals

    # Priority 2 — TREND FOLLOWING
    if abs(trend_direction) > trend_threshold and risk_level <= risk_threshold:
        momentum_feature = enhanced_s[123]  # Assuming this is the momentum feature
        if trend_direction > 0 and momentum_feature > 0:
            reward += 20  # Positive reward for upward trend and BUY
        elif trend_direction < 0 and momentum_feature < 0:
            reward += 20  # Positive reward for downward trend and SELL

    # Priority 3 — SIDEWAYS / MEAN REVERSION
    if abs(trend_direction) < trend_threshold and risk_level < 0.3:
        reward += 10  # Reward mean-reversion features

    # Priority 4 — HIGH VOLATILITY
    if volatility_level > 0.6 * historical_std:
        reward *= 0.5  # Reduce reward magnitude by 50%

    return np.clip(reward, -100, 100)  # Ensure reward is within [-100, 100]